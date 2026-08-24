pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let mut out = vec![0f32; m * n];
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for p in 0..k {
                let a_ip = a[i * k + p];
                if a_ip == 0.0 {
                    continue;
                }
                let b_row = &b[p * n..p * n + n];
                for j in 0..n {
                    row[j] += a_ip * b_row[j];
                }
            }
        });
        out
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            let out_row = &mut out[i * n..i * n + n];
            for p in 0..k {
                let a_ip = a[i * k + p];
                if a_ip == 0.0 {
                    continue;
                }
                let b_row = &b[p * n..p * n + n];
                for j in 0..n {
                    out_row[j] += a_ip * b_row[j];
                }
            }
        }
        out
    }
}
