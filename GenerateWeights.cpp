/*
 * =============================================================================
 * generate_weights.cpp
 * =============================================================================
 *
 * PURPOSE:
 *   Generates synthetic LLM weight tensors drawn from a Gaussian distribution
 *   with sparse element-wise outliers, then produces a "finetuned" variant
 *   via Bayesian posterior resampling. Weights are saved in 7 binary formats:
 *   FP32, FP16, BF16, FP8-E4M3, FP8-E5M2, MXFP4-E2M1, NVFP4-E2M1.
 *
 * =============================================================================
 * DESIGN OVERVIEW
 * =============================================================================
 *
 *  Process/Thread layout (36 virtual cores total):
 *    main()
 *    ├── create all output directories
 *    └── for each variance in VARIANCES[6]:
 *        └── fork() one child process
 *            └── variance_worker()
 *                ├── open temp dir for this variance
 *                ├── divide 2^30 elements into THREADS_PER_PROC chunks
 *                └── spawn THREADS_PER_PROC std::threads
 *                    └── thread_worker()  [see per-element logic below]
 *                after join: concatenate per-thread temp files -> final files
 *                            delete temp files
 *
 *  6 processes x 6 threads = 36 total workers, matching the 36 virtual cores.
 *
 * =============================================================================
 * PER-ELEMENT LOGIC  (executed inside each thread for every element)
 * =============================================================================
 *
 *   Step 1: Draw is_outlier ~ Bernoulli(P_OUTLIER = 0.001)
 *           -> sigma_type = is_outlier ? KAPPA*sigma : sigma
 *
 *   Step 2: Draw w_base ~ N(0, sigma_type^2)
 *           This is the base model weight.
 *
 *   Step 3: Draw do_redraw ~ Bernoulli(P_REDRAW = 0.01)
 *           If do_redraw:
 *               Posterior update: w_redraw ~ N(w_base/2, sigma_type^2/2)
 *               (Conjugate Gaussian posterior: prior N(0,s^2),
 *                likelihood N(w,s^2), posterior mean=w/2, var=s^2/2)
 *           Else:
 *               w_redraw = w_base  (finetuned weight == base weight)
 *
 *   Step 4: Both w_base and w_redraw are pushed into per-format local buffers.
 *           When a buffer reaches the format's block size, it is flushed:
 *             - Unscaled formats (FP32/FP16/BF16): convert + write to temp file
 *             - Scaled formats: compute amax over block, derive scale,
 *               quantize all elements, write scale to scale-temp-file and
 *               quantized bytes to weight-temp-file.
 *
 * =============================================================================
 * FORMAT SPECIFICATIONS
 * =============================================================================
 *
 *   FP32E8M23  : 4 bytes/elem | no scale | write raw IEEE-754 float bits
 *   FP16E5M10  : 2 bytes/elem | no scale | clamp +-65504, round mantissa to 10b
 *   BF16E8M7   : 2 bytes/elem | no scale | same exponent as FP32, mantissa 7b
 *   FP8E4M3    : 1 byte/elem  | per-block FP32 scale (block=128) | max=448
 *                  scale = fmt_max / amax; written as 4-byte float to scales.bin
 *   FP8E5M2    : 1 byte/elem  | per-block FP32 scale (block=128) | max=57344
 *                  same as FP8E4M3 but different bit layout and max
 *   MXFP4E2M1 : 4 bits/elem, packed 2/byte | per-block E8M0 scale (block=32)
 *                  scale = smallest power-of-two >= amax/6
 *                  E8M0 byte = IEEE FP32 biased exponent of that power-of-two
 *   NVFP4E2M1 : 4 bits/elem, packed 2/byte | per-block E4M3 scale (block=16)
 *                  scale stored as FP8-E4M3 byte representing amax/6
 *                  No tensor-level scale (dropped: requires full pass first).
 *
 * =============================================================================
 * FILE LAYOUT
 * =============================================================================
 *
 *   RESULTS/
 *   └── {FormatLabel}/                 e.g. FP32E8M23, FP8E4M3, NVFP4E2M1
 *       └── {VarLabel}/                e.g. Var0p01, Var0p25
 *           ├── Base/
 *           │   ├── weights.bin
 *           │   └── scales.bin         (only for FP8*, MXFP4, NVFP4)
 *           └── Redraw0p01/
 *               ├── weights.bin
 *               └── scales.bin
 *
 *   Temp files during generation:
 *     RESULTS/_tmp_var{v}/{fmt}_{base|redraw}_{weights|scales}_{tid}.tmp
 *   These are concatenated in thread-index order and then deleted.
 *
 * =============================================================================
 * COMPILATION
 * =============================================================================
 *   See CMakeLists.txt. Requires C++17, -lpthread.
 *   Build: mkdir build && cd build && cmake .. && make -j36
 *
 * SLURM: 6 processes x 6 threads = 36 virtual cores. See submit.slurm.
 * =============================================================================
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// CONSTANTS AND ABLATION CONFIGURATION
// =============================================================================

// Total parameters per model (2^30 = 1,073,741,824)
static constexpr size_t N_PARAMS         = (1ULL << 10);

// Fixed parameters (not ablated)
static constexpr double P_OUTLIER        = 0.001;   // 0.1% of elements are outliers
static constexpr double KAPPA            = 20.0;    // outlier sigma multiplier
static constexpr double P_REDRAW         = 0.01;    // 1% of elements redrawn in finetuned

// Parallelism: 6 processes x 6 threads = 36 cores
static constexpr int N_VARIANCES         = 6;
static constexpr int THREADS_PER_PROC    = 8;

// Variance ablation (sigma^2). sigma ranges from 0.1 to ~0.71.
// These span tight (LLM-like, sigma~0.1-0.3) through wide (stress test).
static constexpr double VARIANCES[N_VARIANCES] = {
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5
};

// Hardcoded directory-safe names for each variance (avoids float->string issues)
static const char* VAR_NAMES[N_VARIANCES] = {
    "Var0p01", "Var0p025", "Var0p05", "Var0p1", "Var0p25", "Var0p5"
};

// Block sizes for scaled formats
static constexpr size_t BLOCK_FP8    = 128;  // per-block FP32 scale
static constexpr size_t BLOCK_MXFP4 = 32;   // per-block E8M0 (1-byte power-of-two)
static constexpr size_t BLOCK_NVFP4 = 16;   // per-block E4M3 (FP8 byte)

// Format representable maxima (used for scaling)
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr float FP8_E5M2_MAX = 57344.0f;
static constexpr float FP4_E2M1_MAX = 6.0f;
static constexpr float FP16_MAX     = 65504.0f;

// Output root
static const std::string RESULTS_DIR = "RESULTS";

// Byte output buffer size for unscaled formats (before fwrite)
static constexpr size_t UNBUF = 4096;

// =============================================================================
// FORMAT ENUM AND METADATA
// =============================================================================

enum class Fmt { FP32, FP16, BF16, FP8E4M3, FP8E5M2, MXFP4, NVFP4 };
static constexpr int N_FMTS = 7;

static const Fmt FMTS[N_FMTS] = {
    Fmt::FP32, Fmt::FP16, Fmt::BF16,
    Fmt::FP8E4M3, Fmt::FP8E5M2,
    Fmt::MXFP4, Fmt::NVFP4
};

static const char* FORMAT_NAMES[N_FMTS] = {
    "FP32E8M23",
    "FP16E5M10",
    "BF16E8M7",
    "FP8E4M3",
    "FP8E5M2",
    "MXFP4E2M1",
    "NVFP4E2M1"
};

// Returns true if this format requires a separate scales.bin file
static bool has_scale(Fmt f) {
    return f == Fmt::FP8E4M3 || f == Fmt::FP8E5M2 ||
           f == Fmt::MXFP4   || f == Fmt::NVFP4;
}

// =============================================================================
// FORMAT CONVERSION UTILITIES
// =============================================================================

// ---------------------------------------------------------------------------
// FP16 (E5M10): clamp to +-65504, rebase exponent from FP32 bias 127 to
// FP16 bias 15, truncate mantissa from 23 to 10 bits with round-to-nearest.
// ---------------------------------------------------------------------------
static uint16_t to_fp16(float v) {
    v = std::clamp(v, -FP16_MAX, FP16_MAX);
    uint32_t bits;
    memcpy(&bits, &v, 4);

    uint16_t sign    = (uint16_t)((bits >> 16) & 0x8000u);
    int32_t  exp32   = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t mant32  = bits & 0x7FFFFFu;

    // Handle zero and subnormals: map to signed zero
    if (exp32 == 0) return sign;

    int32_t exp16 = exp32 - 127 + 15;
    if (exp16 <= 0) return sign;           // underflow -> zero
    if (exp16 >= 31) return sign | 0x7BFFu; // overflow -> max finite

    // Round-to-nearest: add guard bit (bit 12 of FP32 mantissa)
    uint32_t rounded = mant32 + (1u << 12);
    uint16_t mant10  = (uint16_t)(rounded >> 13);
    if (mant10 >= (1u << 10)) {
        // Mantissa rounded up into next exponent
        exp16++;
        mant10 = 0;
        if (exp16 >= 31) return sign | 0x7BFFu;
    }
    return sign | ((uint16_t)exp16 << 10) | mant10;
}

// ---------------------------------------------------------------------------
// BF16 (E8M7): shares FP32 exponent exactly; mantissa is top 7 bits.
// Round-to-nearest-even by adding a rounding constant before truncation.
// ---------------------------------------------------------------------------
static uint16_t to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    // Round-to-nearest-even: add 0x7FFF + (bit 16 of original, for "even" tie-break)
    uint32_t rnd = 0x00007FFFu + ((bits >> 16) & 1u);
    bits += rnd;
    return (uint16_t)(bits >> 16);
}

// ---------------------------------------------------------------------------
// FP8 E4M3: bias=7, no infinities, NaN=0x7F, max=448.
// Encodes as: [sign(1) | exp(4) | mant(3)]
// ---------------------------------------------------------------------------
static uint8_t to_fp8_e4m3(float v) {
    v = std::clamp(v, -FP8_E4M3_MAX, FP8_E4M3_MAX);
    if (v == 0.0f) return 0x00u;

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint8_t  sign   = (uint8_t)((bits >> 31) & 1u);
    int32_t  exp32  = (int32_t)((bits >> 23) & 0xFFu) - 127; // unbiased FP32 exp
    uint32_t mant32 = bits & 0x7FFFFFu;

    // Rebias to E4M3 (bias=7)
    int32_t exp8 = exp32 + 7;
    if (exp8 <= 0) return (uint8_t)(sign << 7); // underflow -> signed zero

    // Max normal: exponent 14, mantissa 0b110 = 0x7E (NaN is 0x7F, reserved)
    if (exp8 >= 15) return (uint8_t)((sign << 7) | 0x7Eu);

    // Round-to-nearest: bit 19 is the guard bit for a 3-bit mantissa
    uint32_t mant3 = (mant32 + (1u << 19)) >> 20;
    if (mant3 >= 8u) { exp8++; mant3 = 0; }
    if (exp8 >= 15)   return (uint8_t)((sign << 7) | 0x7Eu);

    return (uint8_t)((sign << 7) | ((uint8_t)exp8 << 3) | (uint8_t)(mant3 & 0x7u));
}

// ---------------------------------------------------------------------------
// FP8 E5M2: bias=15, has infinities, NaN=0x7F/0xFF etc., max=57344.
// Encodes as: [sign(1) | exp(5) | mant(2)]
// ---------------------------------------------------------------------------
static uint8_t to_fp8_e5m2(float v) {
    v = std::clamp(v, -FP8_E5M2_MAX, FP8_E5M2_MAX);
    if (v == 0.0f) return 0x00u;

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint8_t  sign   = (uint8_t)((bits >> 31) & 1u);
    int32_t  exp32  = (int32_t)((bits >> 23) & 0xFFu) - 127;
    uint32_t mant32 = bits & 0x7FFFFFu;

    int32_t exp8 = exp32 + 15;
    if (exp8 <= 0) return (uint8_t)(sign << 7); // underflow -> signed zero

    // Max finite: exponent 30, mantissa 0b11 = 0x7B
    if (exp8 >= 31) return (uint8_t)((sign << 7) | 0x7Bu);

    // Round-to-nearest: guard bit is bit 20 for 2-bit mantissa
    uint32_t mant2 = (mant32 + (1u << 20)) >> 21;
    if (mant2 >= 4u) { exp8++; mant2 = 0; }
    if (exp8 >= 31)   return (uint8_t)((sign << 7) | 0x7Bu);

    return (uint8_t)((sign << 7) | ((uint8_t)exp8 << 2) | (uint8_t)(mant2 & 0x3u));
}

// ---------------------------------------------------------------------------
// E2M1 (FP4): used by both MXFP4 and NVFP4.
// Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// Encoding: [sign(1) | index(3)] where index 0..7 maps to the table above.
// Input v has already been divided by the block scale before calling this.
// ---------------------------------------------------------------------------
static const float E2M1_TABLE[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

static uint8_t to_e2m1(float v) {
    float av = std::fabs(v);
    av = std::clamp(av, 0.0f, FP4_E2M1_MAX);

    // Find nearest representable magnitude by exhaustive search (8 values only)
    int   best      = 0;
    float best_dist = std::fabs(av - E2M1_TABLE[0]);
    for (int i = 1; i < 8; i++) {
        float d = std::fabs(av - E2M1_TABLE[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }

    uint8_t sign = (v < 0.0f) ? 1u : 0u;
    // 4-bit encoding: [sign | e1 | e0 | m0] = [sign | best[2] | best[1] | best[0]]
    return (uint8_t)((sign << 3) | (uint8_t)best);
}

// Pack two 4-bit E2M1 nibbles into one byte.
// Convention: lo nibble = element[i], hi nibble = element[i+1]
static uint8_t pack_e2m1(uint8_t lo, uint8_t hi) {
    return (uint8_t)((hi << 4) | (lo & 0x0Fu));
}

// ---------------------------------------------------------------------------
// E8M0 scale for MXFP4: smallest power-of-two >= (amax / 6).
// Stored as the raw IEEE-754 biased exponent byte of that power-of-two.
// Value 127 -> 2^0 = 1.0. This is safe for amax==0 (returns 127 -> scale=1).
// ---------------------------------------------------------------------------
static uint8_t to_e8m0_scale(float amax) {
    if (amax == 0.0f) return 127u; // 2^(127-127) = 1.0 -> safe divisor, no info lost

    float needed = amax / FP4_E2M1_MAX; // minimum scale s.t. amax/scale <= 6
    // Round up to next power of two: 2^ceil(log2(needed))
    float p = std::pow(2.0f, std::ceil(std::log2(needed)));

    uint32_t bits;
    memcpy(&bits, &p, 4);
    // Return the biased exponent byte from the FP32 representation
    return (uint8_t)((bits >> 23) & 0xFFu);
}

// Decode E8M0 byte back to float: value = 2^(e - 127)
static float from_e8m0(uint8_t e) {
    // Reconstruct FP32: sign=0, exp=e, mant=0 -> value = 1.0 * 2^(e-127)
    uint32_t bits = (uint32_t)e << 23;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

// ---------------------------------------------------------------------------
// E4M3 scale byte for NVFP4: encodes (amax / 6) as an FP8-E4M3 byte.
// Decoded back with decode_e4m3_scale() to get the actual divisor used.
// ---------------------------------------------------------------------------
static uint8_t to_e4m3_scale(float amax) {
    float scale = (amax == 0.0f) ? 1.0f : (amax / FP4_E2M1_MAX);
    return to_fp8_e4m3(scale); // reuse FP8 E4M3 encoder; scale is always >= 0
}

// Decode an E4M3 byte back to float, handling subnormals correctly.
// Used to recover the actual scale divisor applied during NVFP4 quantization.
static float decode_e4m3_scale(uint8_t b) {
    uint8_t sign = (b >> 7) & 1u;
    int32_t exp  = (b >> 3) & 0x0Fu;
    uint8_t mant = b & 0x07u;

    float v;
    if (exp == 0) {
        // Subnormal E4M3: no implicit leading 1; value = (mant/8) * 2^(-6)
        v = (mant / 8.0f) * std::pow(2.0f, -6.0f);
    } else {
        // Normal E4M3: value = (1 + mant/8) * 2^(exp - bias), bias=7
        v = (1.0f + mant / 8.0f) * std::pow(2.0f, (float)(exp - 7));
    }
    return (sign && v != 0.0f) ? -v : v;
}

// =============================================================================
// TEMP FILE NAMING
// =============================================================================

// Returns the temp file path for a given format, model type, part, and thread.
// Pattern: {tmp_dir}/{fmt}_{model}_{part}_{tid}.tmp
static std::string tmp_name(const std::string& dir,
                            const char*         fmt,
                            const char*         model,  // "base" or "redraw"
                            const char*         part,   // "weights" or "scales"
                            int                 tid) {
    return dir + "/" + fmt + "_" + model + "_" + part + "_"
               + std::to_string(tid) + ".tmp";
}

// =============================================================================
// FILE CONCATENATION HELPER
// =============================================================================

// Appends the contents of each file in `parts` (in order) into `dest`,
// then removes each part file.
static void concat_and_delete(const std::string&              dest,
                              const std::vector<std::string>& parts) {
    std::ofstream out(dest, std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] Cannot open output file: " << dest << "\n";
        return;
    }
    for (const auto& p : parts) {
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            std::cerr << "[ERROR] Missing temp file: " << p << "\n";
            continue;
        }
        out << in.rdbuf();
        in.close();
        fs::remove(p);
    }
}

// =============================================================================
// THREAD WORKER CONFIGURATION
// =============================================================================

struct ThreadConfig {
    int         thread_id;  // 0 .. THREADS_PER_PROC-1
    int         var_idx;    // index into VARIANCES[]
    double      sigma;      // sqrt(VARIANCES[var_idx])
    size_t      start;      // first global element index for this thread (unused,
                            // kept for potential future use / documentation)
    size_t      count;      // number of elements this thread generates
    std::string tmp_dir;    // directory for temp files
};

// =============================================================================
// THREAD WORKER
// =============================================================================

static void thread_worker(ThreadConfig cfg) {
    // -------------------------------------------------------------------------
    // RNG: seeded deterministically from (var_idx, thread_id) for
    // reproducibility. Each thread has its own independent generator.
    // -------------------------------------------------------------------------
    uint64_t seed = (uint64_t)cfg.var_idx * 10000ULL + (uint64_t)cfg.thread_id;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double>       norm_dist(0.0, 1.0);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    const double sigma     = cfg.sigma;
    const double sigma_out = KAPPA * sigma; // outlier elements use this sigma

    // -------------------------------------------------------------------------
    // Open one temp file per format x {base, redraw} x {weights, [scales]}.
    // Using raw FILE* for maximum sequential write throughput.
    // -------------------------------------------------------------------------
    FILE* fw_base[N_FMTS]   = {};
    FILE* fw_redraw[N_FMTS] = {};
    FILE* fs_base[N_FMTS]   = {};   // scale files (NULL for unscaled formats)
    FILE* fs_redraw[N_FMTS] = {};

    for (int f = 0; f < N_FMTS; f++) {
        const char* fn  = FORMAT_NAMES[f];
        Fmt         fmt = FMTS[f];

        fw_base[f] = fopen(
            tmp_name(cfg.tmp_dir, fn, "base",   "weights", cfg.thread_id).c_str(), "wb");
        fw_redraw[f] = fopen(
            tmp_name(cfg.tmp_dir, fn, "redraw", "weights", cfg.thread_id).c_str(), "wb");

        if (!fw_base[f] || !fw_redraw[f]) {
            std::cerr << "[ERROR] Cannot open weight temp file for fmt "
                      << fn << " tid=" << cfg.thread_id << "\n";
        }

        if (has_scale(fmt)) {
            fs_base[f] = fopen(
                tmp_name(cfg.tmp_dir, fn, "base",   "scales", cfg.thread_id).c_str(), "wb");
            fs_redraw[f] = fopen(
                tmp_name(cfg.tmp_dir, fn, "redraw", "scales", cfg.thread_id).c_str(), "wb");

            if (!fs_base[f] || !fs_redraw[f]) {
                std::cerr << "[ERROR] Cannot open scale temp file for fmt "
                          << fn << " tid=" << cfg.thread_id << "\n";
            }
        }
    }

    // -------------------------------------------------------------------------
    // Per-format output buffers.
    //
    // Unscaled formats (FP32/FP16/BF16): accumulate converted bytes into a
    // byte buffer of size UNBUF, flush to disk when full.
    //
    // Scaled formats (FP8/MXFP4/NVFP4): accumulate raw FP32 values into a
    // float buffer of size = block_size; when full, compute amax -> scale ->
    // quantize -> write scale file + weight file, then reset.
    // -------------------------------------------------------------------------

    // --- Unscaled byte buffers ---
    uint8_t fp32_base_buf[UNBUF],   fp32_redraw_buf[UNBUF];
    uint8_t fp16_base_buf[UNBUF],   fp16_redraw_buf[UNBUF];
    uint8_t bf16_base_buf[UNBUF],   bf16_redraw_buf[UNBUF];
    size_t  fp32_base_n = 0,        fp32_redraw_n = 0;
    size_t  fp16_base_n = 0,        fp16_redraw_n = 0;
    size_t  bf16_base_n = 0,        bf16_redraw_n = 0;

    // --- Scaled format FP32 accumulation buffers ---
    float fp8e4m3_base_blk[BLOCK_FP8],    fp8e4m3_redraw_blk[BLOCK_FP8];
    float fp8e5m2_base_blk[BLOCK_FP8],    fp8e5m2_redraw_blk[BLOCK_FP8];
    float mxfp4_base_blk[BLOCK_MXFP4],    mxfp4_redraw_blk[BLOCK_MXFP4];
    float nvfp4_base_blk[BLOCK_NVFP4],    nvfp4_redraw_blk[BLOCK_NVFP4];
    size_t fp8e4m3_n = 0, fp8e5m2_n = 0, mxfp4_n = 0, nvfp4_n = 0;

    // -------------------------------------------------------------------------
    // Flush helpers (lambdas) — called when a scaled block buffer is full.
    // Each writes one scale value then N quantized weight bytes/nibble-pairs.
    // -------------------------------------------------------------------------

    // FP8 E4M3 or E5M2 block flush.
    // scale = fmt_max / amax; written as 4-byte FP32 to fs_file.
    // Each element multiplied by scale then quantized to 1 byte.
    auto flush_fp8 = [&](float* blk, size_t sz,
                          float fmt_max, bool is_e4m3,
                          FILE* fw, FILE* fs_file)
    {
        // Compute amax over this block
        float amax = 0.0f;
        for (size_t i = 0; i < sz; i++) amax = std::max(amax, std::fabs(blk[i]));

        // Scale maps amax -> fmt_max; use 1.0 if block is all zeros
        float scale = (amax == 0.0f) ? 1.0f : (fmt_max / amax);

        // Write FP32 scale to scale file (one per block)
        fwrite(&scale, sizeof(float), 1, fs_file);

        // Quantize each element and write to weight file
        for (size_t i = 0; i < sz; i++) {
            float   scaled = blk[i] * scale;
            uint8_t q      = is_e4m3 ? to_fp8_e4m3(scaled) : to_fp8_e5m2(scaled);
            fwrite(&q, 1, 1, fw);
        }
    };

    // MXFP4 block flush.
    // scale = smallest power-of-two >= amax/6, stored as 1-byte E8M0.
    // Elements divided by scale, quantized to E2M1, packed 2-per-byte.
    auto flush_mxfp4 = [&](float* blk, size_t sz,
                             FILE* fw, FILE* fs_file)
    {
        float amax = 0.0f;
        for (size_t i = 0; i < sz; i++) amax = std::max(amax, std::fabs(blk[i]));

        uint8_t e8m0      = to_e8m0_scale(amax);      // E8M0 scale byte
        float   scale_val = from_e8m0(e8m0);           // actual float divisor

        // Write 1-byte E8M0 scale to scale file
        fwrite(&e8m0, 1, 1, fs_file);

        // Quantize and pack pairs of E2M1 nibbles
        // sz is always even (block=32, and we pad to even on partial flush)
        for (size_t i = 0; i < sz; i += 2) {
            float   v0     = blk[i]   / scale_val;
            float   v1     = blk[i+1] / scale_val;
            uint8_t q0     = to_e2m1(v0);
            uint8_t q1     = to_e2m1(v1);
            uint8_t packed = pack_e2m1(q0, q1);
            fwrite(&packed, 1, 1, fw);
        }
    };

    // NVFP4 block flush.
    // scale = amax/6, stored as 1-byte E4M3 (FP8).
    // Elements divided by decoded scale, quantized to E2M1, packed 2-per-byte.
    auto flush_nvfp4 = [&](float* blk, size_t sz,
                             FILE* fw, FILE* fs_file)
    {
        float amax = 0.0f;
        for (size_t i = 0; i < sz; i++) amax = std::max(amax, std::fabs(blk[i]));

        uint8_t e4m3_byte = to_e4m3_scale(amax);         // encode scale as E4M3 byte
        float   scale_val = decode_e4m3_scale(e4m3_byte); // recover actual float divisor
        if (scale_val == 0.0f) scale_val = 1.0f;          // safety: avoid div by zero

        // Write 1-byte E4M3 scale to scale file
        fwrite(&e4m3_byte, 1, 1, fs_file);

        // Quantize and pack pairs of E2M1 nibbles
        for (size_t i = 0; i < sz; i += 2) {
            float   v0     = blk[i]   / scale_val;
            float   v1     = blk[i+1] / scale_val;
            uint8_t q0     = to_e2m1(v0);
            uint8_t q1     = to_e2m1(v1);
            uint8_t packed = pack_e2m1(q0, q1);
            fwrite(&packed, 1, 1, fw);
        }
    };

    // -------------------------------------------------------------------------
    // Helper macro: flush unscaled byte buffer when full
    // -------------------------------------------------------------------------
#define FLUSH_UNBUF(buf, n, fw)                              \
    do {                                                     \
        if ((n) >= UNBUF) {                                  \
            fwrite((buf), 1, (n), (fw));                     \
            (n) = 0;                                         \
        }                                                    \
    } while (0)

    // -------------------------------------------------------------------------
    // MAIN GENERATION LOOP
    // Each iteration generates one (w_base, w_redraw) pair and pushes both
    // through all format conversion pipelines simultaneously.
    // -------------------------------------------------------------------------
    for (size_t idx = 0; idx < cfg.count; idx++) {

        // --- Step 1: Bernoulli draw for outlier status ---
        // Outlier elements use sigma_out = KAPPA * sigma
        bool   is_outlier = (unit_dist(rng) < P_OUTLIER);
        double s          = is_outlier ? sigma_out : sigma;

        // --- Step 2: Draw base weight w_base ~ N(0, s^2) ---
        double w_base = norm_dist(rng) * s;

        // --- Step 3: Bernoulli draw for finetuning redraw ---
        // If redrawn: posterior w_redraw ~ N(w_base/2, s^2/2)
        // Else:       w_redraw = w_base (no change in finetuned model)
        bool   do_redraw = (unit_dist(rng) < P_REDRAW);
        double w_redraw;
        if (do_redraw) {
            double post_mean  = w_base * 0.5;
            double post_sigma = s / std::sqrt(2.0);
            w_redraw = post_mean + norm_dist(rng) * post_sigma;
        } else {
            w_redraw = w_base;
        }

        // Cast to float for all format conversions
        float wb = (float)w_base;
        float wr = (float)w_redraw;

        // =====================================================================
        // FP32: write raw 4-byte IEEE-754 float directly (fmt index 0)
        // =====================================================================
        memcpy(&fp32_base_buf[fp32_base_n],     &wb, 4); fp32_base_n   += 4;
        memcpy(&fp32_redraw_buf[fp32_redraw_n], &wr, 4); fp32_redraw_n += 4;
        FLUSH_UNBUF(fp32_base_buf,   fp32_base_n,   fw_base[0]);
        FLUSH_UNBUF(fp32_redraw_buf, fp32_redraw_n, fw_redraw[0]);

        // =====================================================================
        // FP16: 2 bytes per element, no scale (fmt index 1)
        // =====================================================================
        {
            uint16_t hb = to_fp16(wb), hr = to_fp16(wr);
            memcpy(&fp16_base_buf[fp16_base_n],     &hb, 2); fp16_base_n   += 2;
            memcpy(&fp16_redraw_buf[fp16_redraw_n], &hr, 2); fp16_redraw_n += 2;
            FLUSH_UNBUF(fp16_base_buf,   fp16_base_n,   fw_base[1]);
            FLUSH_UNBUF(fp16_redraw_buf, fp16_redraw_n, fw_redraw[1]);
        }

        // =====================================================================
        // BF16: 2 bytes per element, no scale (fmt index 2)
        // =====================================================================
        {
            uint16_t bb = to_bf16(wb), br = to_bf16(wr);
            memcpy(&bf16_base_buf[bf16_base_n],     &bb, 2); bf16_base_n   += 2;
            memcpy(&bf16_redraw_buf[bf16_redraw_n], &br, 2); bf16_redraw_n += 2;
            FLUSH_UNBUF(bf16_base_buf,   bf16_base_n,   fw_base[2]);
            FLUSH_UNBUF(bf16_redraw_buf, bf16_redraw_n, fw_redraw[2]);
        }

        // =====================================================================
        // FP8 E4M3: accumulate into block of 128, then flush (fmt index 3)
        // =====================================================================
        fp8e4m3_base_blk[fp8e4m3_n]   = wb;
        fp8e4m3_redraw_blk[fp8e4m3_n] = wr;
        fp8e4m3_n++;
        if (fp8e4m3_n == BLOCK_FP8) {
            flush_fp8(fp8e4m3_base_blk,   BLOCK_FP8, FP8_E4M3_MAX, true,
                      fw_base[3],   fs_base[3]);
            flush_fp8(fp8e4m3_redraw_blk, BLOCK_FP8, FP8_E4M3_MAX, true,
                      fw_redraw[3], fs_redraw[3]);
            fp8e4m3_n = 0;
        }

        // =====================================================================
        // FP8 E5M2: accumulate into block of 128, then flush (fmt index 4)
        // =====================================================================
        fp8e5m2_base_blk[fp8e5m2_n]   = wb;
        fp8e5m2_redraw_blk[fp8e5m2_n] = wr;
        fp8e5m2_n++;
        if (fp8e5m2_n == BLOCK_FP8) {
            flush_fp8(fp8e5m2_base_blk,   BLOCK_FP8, FP8_E5M2_MAX, false,
                      fw_base[4],   fs_base[4]);
            flush_fp8(fp8e5m2_redraw_blk, BLOCK_FP8, FP8_E5M2_MAX, false,
                      fw_redraw[4], fs_redraw[4]);
            fp8e5m2_n = 0;
        }

        // =====================================================================
        // MXFP4 E2M1: accumulate into block of 32, then flush (fmt index 5)
        // =====================================================================
        mxfp4_base_blk[mxfp4_n]   = wb;
        mxfp4_redraw_blk[mxfp4_n] = wr;
        mxfp4_n++;
        if (mxfp4_n == BLOCK_MXFP4) {
            flush_mxfp4(mxfp4_base_blk,   BLOCK_MXFP4, fw_base[5],   fs_base[5]);
            flush_mxfp4(mxfp4_redraw_blk, BLOCK_MXFP4, fw_redraw[5], fs_redraw[5]);
            mxfp4_n = 0;
        }

        // =====================================================================
        // NVFP4 E2M1: accumulate into block of 16, then flush (fmt index 6)
        // =====================================================================
        nvfp4_base_blk[nvfp4_n]   = wb;
        nvfp4_redraw_blk[nvfp4_n] = wr;
        nvfp4_n++;
        if (nvfp4_n == BLOCK_NVFP4) {
            flush_nvfp4(nvfp4_base_blk,   BLOCK_NVFP4, fw_base[6],   fs_base[6]);
            flush_nvfp4(nvfp4_redraw_blk, BLOCK_NVFP4, fw_redraw[6], fs_redraw[6]);
            nvfp4_n = 0;
        }

    } // end main generation loop

#undef FLUSH_UNBUF

    // -------------------------------------------------------------------------
    // Flush any remaining data in buffers after the loop.
    // N_PARAMS is a power of two and block sizes all divide 2^24, so partial
    // blocks only occur if count is not a multiple of the block size (possible
    // for the last thread if N_PARAMS % THREADS_PER_PROC != 0).
    // We pad FP4 blocks to even length for nibble packing.
    // -------------------------------------------------------------------------

    // Unscaled remainders
    if (fp32_base_n   > 0) fwrite(fp32_base_buf,   1, fp32_base_n,   fw_base[0]);
    if (fp32_redraw_n > 0) fwrite(fp32_redraw_buf, 1, fp32_redraw_n, fw_redraw[0]);
    if (fp16_base_n   > 0) fwrite(fp16_base_buf,   1, fp16_base_n,   fw_base[1]);
    if (fp16_redraw_n > 0) fwrite(fp16_redraw_buf, 1, fp16_redraw_n, fw_redraw[1]);
    if (bf16_base_n   > 0) fwrite(bf16_base_buf,   1, bf16_base_n,   fw_base[2]);
    if (bf16_redraw_n > 0) fwrite(bf16_redraw_buf, 1, bf16_redraw_n, fw_redraw[2]);

    // FP8 E4M3 remainder
    if (fp8e4m3_n > 0) {
        flush_fp8(fp8e4m3_base_blk,   fp8e4m3_n, FP8_E4M3_MAX, true,
                  fw_base[3],   fs_base[3]);
        flush_fp8(fp8e4m3_redraw_blk, fp8e4m3_n, FP8_E4M3_MAX, true,
                  fw_redraw[3], fs_redraw[3]);
    }

    // FP8 E5M2 remainder
    if (fp8e5m2_n > 0) {
        flush_fp8(fp8e5m2_base_blk,   fp8e5m2_n, FP8_E5M2_MAX, false,
                  fw_base[4],   fs_base[4]);
        flush_fp8(fp8e5m2_redraw_blk, fp8e5m2_n, FP8_E5M2_MAX, false,
                  fw_redraw[4], fs_redraw[4]);
    }

    // MXFP4 remainder: pad to even for nibble packing
    if (mxfp4_n > 0) {
        if (mxfp4_n % 2 != 0) {
            // Pad with zero (contributes 0-nibble, does not affect scale)
            mxfp4_base_blk[mxfp4_n]   = 0.0f;
            mxfp4_redraw_blk[mxfp4_n] = 0.0f;
            mxfp4_n++;
        }
        flush_mxfp4(mxfp4_base_blk,   mxfp4_n, fw_base[5],   fs_base[5]);
        flush_mxfp4(mxfp4_redraw_blk, mxfp4_n, fw_redraw[5], fs_redraw[5]);
    }

    // NVFP4 remainder: pad to even for nibble packing
    if (nvfp4_n > 0) {
        if (nvfp4_n % 2 != 0) {
            nvfp4_base_blk[nvfp4_n]   = 0.0f;
            nvfp4_redraw_blk[nvfp4_n] = 0.0f;
            nvfp4_n++;
        }
        flush_nvfp4(nvfp4_base_blk,   nvfp4_n, fw_base[6],   fs_base[6]);
        flush_nvfp4(nvfp4_redraw_blk, nvfp4_n, fw_redraw[6], fs_redraw[6]);
    }

    // Close all open file handles
    for (int f = 0; f < N_FMTS; f++) {
        if (fw_base[f])   fclose(fw_base[f]);
        if (fw_redraw[f]) fclose(fw_redraw[f]);
        if (fs_base[f])   fclose(fs_base[f]);
        if (fs_redraw[f]) fclose(fs_redraw[f]);
    }
}

// =============================================================================
// VARIANCE WORKER
// Runs in a child process (one per variance). Spawns THREADS_PER_PROC threads,
// waits for them, then concatenates per-thread temp files into final outputs.
// =============================================================================

static void variance_worker(int var_idx) {
    const double sigma = std::sqrt(VARIANCES[var_idx]);
    const char*  vname = VAR_NAMES[var_idx];

    // Temp directory: one per variance process, isolated from other processes
    std::string tmp_dir = RESULTS_DIR + "/_tmp_var" + std::to_string(var_idx);
    fs::create_directories(tmp_dir);

    std::cout << "[var " << var_idx << " / " << vname << "] "
              << "sigma=" << sigma << " spawning "
              << THREADS_PER_PROC << " threads\n" << std::flush;

    // Divide N_PARAMS among threads: each thread gets an equal contiguous chunk.
    // Last thread absorbs any remainder (N_PARAMS is a power of two and
    // THREADS_PER_PROC=6 does not divide it evenly: 2^30 / 6 = 178956970 r4).
    size_t chunk = N_PARAMS / THREADS_PER_PROC;

    std::vector<std::thread> threads;
    threads.reserve(THREADS_PER_PROC);

    for (int t = 0; t < THREADS_PER_PROC; t++) {
        ThreadConfig cfg;
        cfg.thread_id = t;
        cfg.var_idx   = var_idx;
        cfg.sigma     = sigma;
        cfg.start     = (size_t)t * chunk;
        cfg.count     = (t == THREADS_PER_PROC - 1)
                        ? (N_PARAMS - (size_t)t * chunk)  // last thread: remainder
                        : chunk;
        cfg.tmp_dir   = tmp_dir;
        threads.emplace_back(thread_worker, cfg);
    }

    // Wait for all threads to finish generating and writing their temp files
    for (auto& th : threads) th.join();

    std::cout << "[var " << var_idx << "] threads done, concatenating...\n"
              << std::flush;

    // -------------------------------------------------------------------------
    // Concatenate per-thread temp files in thread-index order into final files.
    // Thread 0's chunk comes first, thread 1's next, etc. — matching the
    // sequential element ordering of the full 2^30 parameter array.
    // -------------------------------------------------------------------------
    for (int f = 0; f < N_FMTS; f++) {
        const char* fn  = FORMAT_NAMES[f];
        Fmt         fmt = FMTS[f];

        // Collect temp file paths in thread order
        std::vector<std::string> w_base_parts, w_redraw_parts;
        std::vector<std::string> s_base_parts, s_redraw_parts;

        for (int t = 0; t < THREADS_PER_PROC; t++) {
            w_base_parts.push_back(  tmp_name(tmp_dir, fn, "base",   "weights", t));
            w_redraw_parts.push_back(tmp_name(tmp_dir, fn, "redraw", "weights", t));
            if (has_scale(fmt)) {
                s_base_parts.push_back(  tmp_name(tmp_dir, fn, "base",   "scales", t));
                s_redraw_parts.push_back(tmp_name(tmp_dir, fn, "redraw", "scales", t));
            }
        }

        // Final output directories (already created by main before forking)
        std::string base_dir   = RESULTS_DIR + "/" + fn + "/" + vname + "/Base";
        std::string redraw_dir = RESULTS_DIR + "/" + fn + "/" + vname + "/Redraw0p01";

        concat_and_delete(base_dir   + "/weights.bin", w_base_parts);
        concat_and_delete(redraw_dir + "/weights.bin", w_redraw_parts);

        if (has_scale(fmt)) {
            concat_and_delete(base_dir   + "/scales.bin", s_base_parts);
            concat_and_delete(redraw_dir + "/scales.bin", s_redraw_parts);
        }
    }

    // Remove the now-empty temp directory
    fs::remove_all(tmp_dir);

    std::cout << "[var " << var_idx << " / " << vname << "] DONE\n" << std::flush;
}

// =============================================================================
// DIRECTORY CREATION
// Creates the full RESULTS directory tree before any forking occurs.
// =============================================================================

static void create_all_directories() {
    for (int f = 0; f < N_FMTS; f++) {
        for (int v = 0; v < N_VARIANCES; v++) {
            std::string base_dir =
                std::string(RESULTS_DIR) + "/" + FORMAT_NAMES[f] + "/"
                + VAR_NAMES[v] + "/Base";
            std::string redraw_dir =
                std::string(RESULTS_DIR) + "/" + FORMAT_NAMES[f] + "/"
                + VAR_NAMES[v] + "/Redraw0p01";

            fs::create_directories(base_dir);
            fs::create_directories(redraw_dir);
        }
    }
    std::cout << "All output directories created under " << RESULTS_DIR << "/\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    // Print run configuration for log visibility
    std::cout << "=== generate_weights ===\n"
              << "N_PARAMS      = " << N_PARAMS << " (2^30)\n"
              << "P_OUTLIER     = " << P_OUTLIER << "\n"
              << "KAPPA         = " << KAPPA     << "\n"
              << "P_REDRAW      = " << P_REDRAW  << "\n"
              << "N_VARIANCES   = " << N_VARIANCES << "\n"
              << "THREADS/PROC  = " << THREADS_PER_PROC << "\n"
              << "========================\n\n";

    // Step 1: create all output directories before forking
    // (fork() duplicates the process; creating dirs afterwards would race)
    create_all_directories();

    // Step 2: fork one child process per variance
    std::vector<pid_t> pids;
    pids.reserve(N_VARIANCES);

    for (int v = 0; v < N_VARIANCES; v++) {
        pid_t pid = fork();

        if (pid < 0) {
            std::cerr << "[FATAL] fork() failed for variance index " << v << "\n";
            // Kill already-spawned children before exiting
            for (pid_t p : pids) kill(p, SIGTERM);
            return 1;
        }

        if (pid == 0) {
            // ---- CHILD PROCESS ----
            // Each child handles exactly one variance value.
            variance_worker(v);
            exit(0);
        } else {
            // ---- PARENT PROCESS ----
            pids.push_back(pid);
            std::cout << "Spawned PID " << pid
                      << " for variance=" << VARIANCES[v]
                      << " (" << VAR_NAMES[v] << ")\n" << std::flush;
        }
    }

    // Step 3: wait for all child processes to complete
    int all_ok = 1;
    for (pid_t pid : pids) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::cerr << "[ERROR] Child PID " << pid << " failed (status="
                      << status << ")\n";
            all_ok = 0;
        }
    }

    if (all_ok) {
        std::cout << "\nAll variance workers completed successfully.\n";
    } else {
        std::cerr << "\nOne or more variance workers reported errors.\n";
        return 1;
    }

    return 0;
}