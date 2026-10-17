/* TOPS-class integer kernels for Lumina on ARMv8.2+ (dotprod).
 *
 * Why this file exists: a feasibility study on the target device measured that
 * fp32 BLAS GEMV is memory-bound (84-97% of streaming bandwidth) and that an
 * int8 dotprod GEMV is 3-6x faster than fp32 BLAS GEMV on the real decode
 * shape (11008x4096), while the shipped scalar (numba) integer GEMV measures
 * 0.54-0.89x BLAS. The win only materializes with SIMD integer arithmetic.
 *
 * Kernels (all row-major, framework (N,K) weight layout):
 *   tops_gemv_int8    int8 weights, per-row scale, per-32 activation scales
 *   tops_gemv_q4_0    4-bit GGUF Q4_0 weights, per-group scales (the shipped
 *                     format: 4-bit storage + integer compute in one kernel)
 *   tops_gemv_fp16    fp16 weights/activations (asimdhp)
 *   tops_gemm_int8    int8 GEMM for M>1 (prefill); pthread row-parallel when
 *                     compiled with TOPS_USE_PTHREAD
 *
 * Build (from tops.py): gcc -O3 -shared -fPIC -march=armv8.2-a+dotprod+fp16
 * The caller verifies /proc/cpuinfo has asimddp and runs a numeric self-test
 * before trusting any result; on any failure it falls back to the numba path.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <arm_neon.h>

#ifdef TOPS_USE_PTHREAD
#include <pthread.h>
#endif

#define TOPS_VERSION 4

int tops_version(void) { return TOPS_VERSION; }

/* thread count for the _mt kernels (set once by the caller; default 8) */
static int g_threads = 8;
void tops_set_threads(int n) { g_threads = n > 0 ? n : 1; }
int tops_get_threads(void) { return g_threads; }

/* ---- int8 dotprod GEMV -------------------------------------------------- */
/* out[n] = ws[n] * sum_b (int8 dot of 32-block b) * xs[b]                  */
void tops_gemv_int8(const int8_t *xq, const float *xs, const int8_t *wq,
                    const float *ws, float *out, int N, int K) {
    const int nb = K >> 5;              /* 32-element blocks */
    for (int n = 0; n < N; n++) {
        const int8_t *row = wq + (size_t)n * K;
        const float s = ws[n];
        float acc = 0.0f;
        int32_t tail = 0;
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = xq + (b << 5);
            const int8_t *wb = row + (b << 5);
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            int8x16_t a0 = vld1q_s8(xb);
            int8x16_t b0 = vld1q_s8(wb);
            int8x16_t a1 = vld1q_s8(xb + 16);
            int8x16_t b1 = vld1q_s8(wb + 16);
            d0 = vdotq_s32(d0, a0, b0);
            d1 = vdotq_s32(d1, a1, b1);
            acc += (float)(vaddvq_s32(d0) + vaddvq_s32(d1)) * xs[b];
        }
        /* scalar tail for K not a multiple of 32 (callers normally pad) */
        for (int k = nb << 5; k < K; k++)
            tail += (int32_t)xq[k] * (int32_t)row[k];
        out[n] = (acc + (float)tail) * s;
    }
}

/* ---- Q4_0 (GGUF) dotprod GEMV ------------------------------------------ */
/* wq is the PACKED payload: (N, K/2) bytes, byte j holds weights 2j (low     */
/* nibble) and 2j+1 (high nibble), code 0..15 (value = code - 8).            */
/* ws is (N, K/32) per-group scales. Nibble unpack via two 16-entry tables    */
/* plus interleave: 3 vector ops / 16 MACs, 4-bit storage + integer compute.  */
void tops_gemv_q4_0(const int8_t *xq, const float *xs, const uint8_t *wq,
                    const float *ws, float *out, int N, int K) {
    static const int8_t TBL[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                   0, 1, 2, 3, 4, 5, 6, 7};
    const int8x16_t t = vld1q_s8(TBL);
    const uint8x16_t mask15 = vdupq_n_u8(15);
    const int nb = K >> 5;                    /* 32 weights per block */
    const size_t rowstride = (size_t)(K >> 1); /* packed bytes per row */
    for (int n = 0; n < N; n++) {
        const uint8_t *row = wq + (size_t)n * rowstride;
        const float *wrow = ws + (size_t)n * nb;
        float acc = 0.0f;
        int32_t tail = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *pb = row + (b << 4);   /* 16 bytes = 32 weights */
            const int8_t *xb = xq + (b << 5);
            int32x4_t d = vdupq_n_s32(0);
            uint8x16_t bytes = vld1q_u8(pb);
            int8x16_t lo = vqtbl1q_s8(t, vandq_u8(bytes, mask15));
            int8x16_t hi = vqtbl1q_s8(t, vshrq_n_u8(bytes, 4));
            int8x16_t c0 = vzip1q_s8(lo, hi);
            int8x16_t c1 = vzip2q_s8(lo, hi);
            d = vdotq_s32(d, vld1q_s8(xb), c0);
            d = vdotq_s32(d, vld1q_s8(xb + 16), c1);
            acc += (float)vaddvq_s32(d) * xs[b] * wrow[b];
        }
        for (int k = nb << 5; k < K; k++) {
            uint8_t byte = row[k >> 1];
            int code = (k & 1) ? (byte >> 4) : (byte & 15);
            tail += (int32_t)xq[k] * (int32_t)(code - 8);
        }
        out[n] = acc + (float)tail;
    }
}

/* ---- Q4_0 v2: strided-dot unpack (no interleave) -----------------------
 * dot(x, interleave(lo, hi)) == dot(x_even, lo) + dot(x_odd, hi), and x_even /
 * x_odd are computed once per call instead of per row: 8-wide table lookups
 * plus 8-wide dotprod, no zip, fewer vector ops per 32 MACs. */
void tops_gemv_q4_0_v2(const int8_t *xq, const float *xs, const uint8_t *wq,
                       const float *ws, float *out, int N, int K) {
    static const int8_t TBL[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                   0, 1, 2, 3, 4, 5, 6, 7};
    int8x8x2_t t2;                       /* 16-entry table = 2 registers */
    t2.val[0] = vld1_s8(TBL);
    t2.val[1] = vld1_s8(TBL + 8);
    const uint8x8_t mask8 = vdup_n_u8(15);
    const int nb = K >> 5;
    const size_t rowstride = (size_t)(K >> 1);
    int8_t *xe = (int8_t *)malloc((size_t)(K / 2) + 16);
    int8_t *xo = (int8_t *)malloc((size_t)(K / 2) + 16);
    if (!xe || !xo) { free(xe); free(xo); return; }
    for (int k = 0; k < K; k += 2) { xe[k >> 1] = xq[k]; xo[k >> 1] = xq[k + 1]; }
    for (int n = 0; n < N; n++) {
        const uint8_t *row = wq + (size_t)n * rowstride;
        const float *wrow = ws + (size_t)n * nb;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *pb = row + (b << 4);   /* 16 bytes = 32 weights */
            const int8_t *xev = xe + (b << 4);
            const int8_t *xod = xo + (b << 4);
            int32x2_t d2 = vdup_n_s32(0);
            for (int j = 0; j < 16; j += 8) {
                uint8x8_t bytes = vld1_u8(pb + j);
                int8x8_t lo = vtbl2_s8(t2, vreinterpret_s8_u8(vand_u8(bytes, mask8)));
                int8x8_t hi = vtbl2_s8(t2, vreinterpret_s8_u8(vshr_n_u8(bytes, 4)));
                d2 = vdot_s32(d2, vld1_s8(xev + j), lo);
                d2 = vdot_s32(d2, vld1_s8(xod + j), hi);
            }
            int32_t seg = vget_lane_s32(d2, 0) + vget_lane_s32(d2, 1);
            acc += (float)seg * xs[b] * wrow[b];
        }
        out[n] = acc;
    }
    free(xe);
    free(xo);
}

/* ---- multithreaded row-split GEMV -------------------------------------- */
typedef struct {
    const int8_t *xq;
    const float *xs;
    const int8_t *wq;
    const float *ws;
    float *out;
    int N, K, row0, row1;
} gemv_job;

static void gemv_int8_rows(const int8_t *xq, const float *xs,
                           const int8_t *wq, const float *ws, float *out,
                           int N, int K, int row0, int row1) {
    const int nb = K >> 5;
    for (int n = row0; n < row1; n++) {
        const int8_t *row = wq + (size_t)n * K;
        const float s = ws[n];
        float acc = 0.0f;
        int32_t tail = 0;
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = xq + (b << 5);
            const int8_t *wb = row + (b << 5);
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            int8x16_t a0 = vld1q_s8(xb);
            int8x16_t b0 = vld1q_s8(wb);
            int8x16_t a1 = vld1q_s8(xb + 16);
            int8x16_t b1 = vld1q_s8(wb + 16);
            d0 = vdotq_s32(d0, a0, b0);
            d1 = vdotq_s32(d1, a1, b1);
            acc += (float)(vaddvq_s32(d0) + vaddvq_s32(d1)) * xs[b];
        }
        for (int k = nb << 5; k < K; k++)
            tail += (int32_t)xq[k] * (int32_t)row[k];
        out[n] = (acc + (float)tail) * s;
    }
}

#ifdef TOPS_USE_PTHREAD
static void *gemv_int8_thread(void *arg) {
    gemv_job *j = (gemv_job *)arg;
    gemv_int8_rows(j->xq, j->xs, j->wq, j->ws, j->out, j->N, j->K, j->row0,
                   j->row1);
    return NULL;
}
#endif

void tops_gemv_int8_mt(const int8_t *xq, const float *xs, const int8_t *wq,
                       const float *ws, float *out, int N, int K) {
#ifdef TOPS_USE_PTHREAD
    int T = N < g_threads ? N : g_threads;
    if (T > 1) {
        pthread_t th[64];
        gemv_job jobs[64];
        int per = (N + T - 1) / T;
        int started = 0;
        for (int i = 0; i < T; i++) {
            int r0 = i * per, r1 = (i == T - 1) ? N : (r0 + per);
            if (r1 <= r0) continue;
            jobs[started] = (gemv_job){xq, xs, wq, ws, out, N, K, r0, r1};
            if (pthread_create(&th[started], NULL, gemv_int8_thread,
                               &jobs[started]) != 0) {
                gemv_int8_rows(xq, xs, wq, ws, out, N, K, r0, N);
                return;
            }
            started++;
        }
        for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
        return;
    }
#endif
    gemv_int8_rows(xq, xs, wq, ws, out, N, K, 0, N);
}

/* ---- fp16 GEMV ---------------------------------------------------------- */
void tops_gemv_fp16(const void *x, const void *w, float *out, int N, int K) {
    const __fp16 *xp = (const __fp16 *)x;
    const __fp16 *wp = (const __fp16 *)w;
    for (int n = 0; n < N; n++) {
        const __fp16 *row = wp + (size_t)n * K;
        float16x8_t acc = vdupq_n_f16((__fp16)0);
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            float16x8_t a = vld1q_f16(xp + k);
            float16x8_t b = vld1q_f16(row + k);
            acc = vfmaq_f16(acc, a, b);
        }
        __fp16 s = 0;
        for (int j = 0; j < 8; j++) s += acc[j];
        float tail = 0.0f;
        for (; k < K; k++) tail += (float)(xp[k] * row[k]);
        out[n] = (float)s + tail;
    }
}

/* ---- int8 GEMM (M rows) ------------------------------------------------- */
typedef struct {
    const int8_t *xq;
    const float *xs;
    const int8_t *wq;
    const float *ws;
    float *out;
    int M, N, K, row0, row1;
} gemm_job;

/* ---- int8 GEMM (M rows) -------------------------------------------------
 * M-blocked: the weight row block is loaded once per (m-block, n, k-block) and
 * reused across the block, cutting weight traffic by the block size (the
 * matrix is re-read once per row otherwise, which is the bandwidth floor). */
#define GEMM_MB 16

static void gemm_rows(const int8_t *xq, const float *xs, const int8_t *wq,
                      const float *ws, float *out, int M, int N, int K,
                      int row0, int row1) {
    const int nb = K >> 5;
    for (int mb = row0; mb < row1; mb += GEMM_MB) {
        const int mb_end = (mb + GEMM_MB < row1) ? (mb + GEMM_MB) : row1;
        const int nrows = mb_end - mb;
        for (int n = 0; n < N; n++) {
            const int8_t *row = wq + (size_t)n * K;
            float acc[GEMM_MB];
            for (int i = 0; i < nrows; i++) acc[i] = 0.0f;
            for (int b = 0; b < nb; b++) {
                int8_t blk[32];
                for (int j = 0; j < 32; j++) blk[j] = row[(b << 5) + j];
                const int8x16_t b0 = vld1q_s8(blk);
                const int8x16_t b1 = vld1q_s8(blk + 16);
                for (int i = 0; i < nrows; i++) {
                    const int8_t *xr = xq + (size_t)(mb + i) * K + (b << 5);
                    int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xr), b0);
                    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(xr + 16), b1);
                    acc[i] += (float)(vaddvq_s32(d0) + vaddvq_s32(d1))
                              * xs[(size_t)(mb + i) * nb + b];
                }
            }
            for (int i = 0; i < nrows; i++)
                out[(size_t)(mb + i) * N + n] = acc[i] * ws[n];
        }
    }
}

#ifdef TOPS_USE_PTHREAD
static void *gemm_thread(void *arg) {
    gemm_job *j = (gemm_job *)arg;
    gemm_rows(j->xq, j->xs, j->wq, j->ws, j->out, j->M, j->N, j->K, j->row0,
              j->row1);
    return NULL;
}
#endif

void tops_gemm_int8(const int8_t *xq, const float *xs, const int8_t *wq,
                    const float *ws, float *out, int M, int N, int K) {
#ifdef TOPS_USE_PTHREAD
    int nthreads = M < 8 ? M : 8;
    if (nthreads > 1) {
        pthread_t th[8];
        gemm_job jobs[8];
        int per = (M + nthreads - 1) / nthreads;
        int started = 0;
        for (int i = 0; i < nthreads; i++) {
            int r0 = i * per, r1 = (i == nthreads - 1) ? M : (r0 + per);
            if (r1 <= r0) continue;
            jobs[started] = (gemm_job){xq, xs, wq, ws, out, M, N, K, r0, r1};
            if (pthread_create(&th[started], NULL, gemm_thread,
                               &jobs[started]) != 0) {
                /* serial fallback for the remainder */
                gemm_rows(xq, xs, wq, ws, out, M, N, K, r0, M);
                return;
            }
            started++;
        }
        for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
        return;
    }
#endif
    gemm_rows(xq, xs, wq, ws, out, M, N, K, 0, M);
}
