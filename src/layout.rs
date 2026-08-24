pub(crate) fn transpose_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    swap_leading_axes(data, rows, cols, 1)
}

pub(crate) fn swap_leading_axes(data: &[f32], first: usize, second: usize, tail: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for i in 0..first {
        for j in 0..second {
            let src = (i * second + j) * tail;
            let dst = (j * first + i) * tail;
            out[dst..dst + tail].copy_from_slice(&data[src..src + tail]);
        }
    }
    out
}
