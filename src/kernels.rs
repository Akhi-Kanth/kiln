//! Matrix-vector kernels. Decoding one token at a time is a stream of matvecs,
//! so these loops (and how many bytes of weights they read) set the speed.
//!
//! Weights are row-major `[rows, cols]` (PyTorch `nn.Linear` layout): `out[r] = W[r] · x`.

use rayon::prelude::*;

/// Q8 block size: every 32 weights share one f32 scale (same layout idea as llama.cpp's Q8_0).
pub const QK: usize = 32;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Data,
}

enum Data {
    F32(Vec<f32>),
    /// `q[i] * d[i / QK]` reconstructs weight `i`.
    Q8 {
        q: Vec<i8>,
        d: Vec<f32>,
    },
}

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data: Data::F32(data),
        }
    }

    /// Symmetric int8 quantization, 4x smaller than f32. Rows that don't split
    /// into whole blocks stay f32.
    pub fn quantize(self) -> Self {
        let (Data::F32(w), true) = (&self.data, self.cols.is_multiple_of(QK)) else {
            return self;
        };
        let (q, d) = quantize_q8(w);
        Matrix {
            data: Data::Q8 { q, d },
            ..self
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.data, Data::Q8 { .. })
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        let c = self.cols;
        match &self.data {
            Data::F32(w) => out
                .par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(|(r, o)| *o = dot(&w[r * c..][..c], x)),
            Data::Q8 { q, d } => {
                // Quantize the activation too, so the inner loop is int8 x int8.
                let (xq, xd) = quantize_q8(x);
                let nb = c / QK;
                out.par_iter_mut()
                    .enumerate()
                    .with_min_len(64)
                    .for_each(|(r, o)| *o = dot_q8(&q[r * c..][..c], &d[r * nb..][..nb], &xq, &xd));
            }
        }
    }
}

pub fn quantize_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(x.len() % QK, 0);
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0f32; x.len() / QK];
    for ((xb, qb), d) in x.chunks_exact(QK).zip(q.chunks_exact_mut(QK)).zip(&mut d) {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        *d = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (q, v) in qb.iter_mut().zip(xb) {
            *q = (v * inv).round() as i8;
        }
    }
    (q, d)
}

#[cfg(target_arch = "aarch64")]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(a.len(), b.len());
    let n = a.len() / 16 * 16;
    // SAFETY: NEON is baseline on aarch64; every load is within [0, n) of both slices.
    let mut s = unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        // 4 independent accumulators hide FMA latency.
        let mut acc = [vdupq_n_f32(0.0); 4];
        for i in (0..n).step_by(16) {
            for (k, acc) in acc.iter_mut().enumerate() {
                *acc = vfmaq_f32(
                    *acc,
                    vld1q_f32(pa.add(i + 4 * k)),
                    vld1q_f32(pb.add(i + 4 * k)),
                );
            }
        }
        vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ))
    };
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
fn dot_q8(wq: &[i8], wd: &[f32], xq: &[i8], xd: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert!(wq.len() == xq.len() && wq.len() == wd.len() * QK && wd.len() == xd.len());
    // SAFETY: block b covers [b*32, b*32+32), inside both slices by the assert above.
    unsafe {
        // Keep the running sum in a vector register: one horizontal add per row, not per block.
        let mut sum = vdupq_n_f32(0.0);
        for b in 0..wd.len() {
            let (w, x) = (wq.as_ptr().add(b * QK), xq.as_ptr().add(b * QK));
            let (w0, w1, x0, x1) = (
                vld1q_s8(w),
                vld1q_s8(w.add(16)),
                vld1q_s8(x),
                vld1q_s8(x.add(16)),
            );
            let acc = sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1);
            sum = vfmaq_n_f32(
                sum,
                vcvtq_f32_s32(acc),
                *wd.get_unchecked(b) * *xd.get_unchecked(b),
            );
        }
        vaddvq_f32(sum)
    }
}

/// acc[i] += sum of a[4i..4i+4] * b[4i..4i+4]: 16 int8 multiply-adds in one instruction.
/// `vdotq_s32` is still unstable in Rust, so emit SDOT directly.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline(always)]
unsafe fn sdot(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    std::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags));
    acc
}

/// Same result without the dot-product extension: widening multiply, then pairwise add.
#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
#[inline(always)]
unsafe fn sdot(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // i8*i8 fits in i16 since |q| <= 127.
    vpadalq_s16(
        vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b))),
        vmull_high_s8(a, b),
    )
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_q8(wq: &[i8], wd: &[f32], xq: &[i8], xd: &[f32]) -> f32 {
    let blocks = wq.chunks_exact(QK).zip(xq.chunks_exact(QK));
    blocks
        .zip(wd.iter().zip(xd))
        .map(|((w, x), (dw, dx))| {
            w.iter()
                .zip(x)
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum::<i32>() as f32
                * dw
                * dx
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_matvec_close_to_f32() {
        let (rows, cols) = (37, 96);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        Matrix::new(w.clone(), rows, cols).matvec(&x, &mut a);
        Matrix::new(w, rows, cols).quantize().matvec(&x, &mut b);
        for r in 0..rows {
            let exact: f32 = (0..cols)
                .map(|c| ((r * cols + c) * 7919 % 1000) as f32 / 500.0 - 1.0)
                .zip(&x)
                .map(|(w, x)| w * x)
                .sum();
            assert!(
                (a[r] - exact).abs() < 1e-3,
                "f32 row {r}: {} vs {exact}",
                a[r]
            );
            assert!(
                (b[r] - exact).abs() < 0.1,
                "q8 row {r}: {} vs {exact}",
                b[r]
            );
        }
    }
}
