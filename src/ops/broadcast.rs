pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = *a.iter().rev().nth(i).unwrap_or(&1);
        let db = *b.iter().rev().nth(i).unwrap_or(&1);
        assert!(
            da == db || da == 1 || db == 1,
            "cannot broadcast shapes {:?} and {:?}",
            a,
            b
        );
        out[rank - 1 - i] = da.max(db);
    }
    out
}

fn strides_row_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_strides(shape: &[usize], target: &[usize]) -> Vec<usize> {
    let own = strides_row_major(shape);
    let rank = target.len();
    let offset = rank - shape.len();
    let mut strides = vec![0usize; rank];
    for i in 0..shape.len() {
        if shape[i] != 1 {
            strides[offset + i] = own[i];
        }
    }
    strides
}

pub(crate) fn broadcast_binary(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    if a_shape == out_shape && b_shape == out_shape {
        return a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect();
    }

    let a_strides = broadcast_strides(a_shape, out_shape);
    let b_strides = broadcast_strides(b_shape, out_shape);
    let out_strides = strides_row_major(out_shape);
    let n: usize = out_shape.iter().product();
    let rank = out_shape.len();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut rem = i;
        let mut ai = 0usize;
        let mut bi = 0usize;
        for d in 0..rank {
            let stride = out_strides[d].max(1);
            let c = rem / stride;
            rem %= stride;
            ai += c * a_strides[d];
            bi += c * b_strides[d];
        }
        out.push(f(a[ai], b[bi]));
    }
    out
}

pub(crate) fn reduce_to_shape(grad: &[f32], grad_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    if grad_shape == target_shape {
        return grad.to_vec();
    }

    let rank = grad_shape.len();
    let offset = rank - target_shape.len();
    let grad_strides = strides_row_major(grad_shape);
    let out_strides = strides_row_major(target_shape);
    let mut out = vec![0f32; target_shape.iter().product::<usize>().max(1)];
    let n = grad.len();

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let mut rem = i;
        let mut out_idx = 0usize;
        for d in 0..rank {
            let stride = grad_strides[d].max(1);
            let c = rem / stride;
            rem %= stride;
            if d >= offset {
                let j = d - offset;
                if target_shape[j] != 1 {
                    out_idx += c * out_strides[j];
                }
            }
        }
        out[out_idx] += grad[i];
    }
    out
}
