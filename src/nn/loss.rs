use crate::tensor::Tensor;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    diff.mul(&diff).mean()
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let shape = logits.shape();
    assert_eq!(shape.len(), 2, "cross_entropy_loss: expected [batch, classes], got {:?}", shape);
    let (batch, classes) = (shape[0], shape[1]);
    assert_eq!(targets.len(), batch, "cross_entropy_loss: {} targets for {} rows", targets.len(), batch);

    let data = logits.data();
    let mut log_probs = vec![0f32; data.len()];
    let mut total_loss = 0f32;
    for b in 0..batch {
        let row = &data[b * classes..b * classes + classes];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let log_sum_exp = sum_exp.ln() + max;
        for c in 0..classes {
            log_probs[b * classes + c] = row[c] - log_sum_exp;
        }
        total_loss += -log_probs[b * classes + targets[b]];
    }
    let loss = total_loss / batch as f32;

    let targets = targets.to_vec();
    Tensor::make(vec![loss], vec![1], vec![logits.clone()], move |grad_out| {
        let scale = grad_out[0] / batch as f32;
        let mut logits_grad = vec![0f32; batch * classes];
        for b in 0..batch {
            for c in 0..classes {
                let softmax_c = log_probs[b * classes + c].exp();
                let indicator = if c == targets[b] { 1.0 } else { 0.0 };
                logits_grad[b * classes + c] = scale * (softmax_c - indicator);
            }
        }
        vec![logits_grad]
    })
}

pub fn accuracy(logits: &Tensor, targets: &[usize]) -> f32 {
    let shape = logits.shape();
    let (batch, classes) = (shape[0], shape[1]);
    let data = logits.data();
    let mut correct = 0usize;
    for b in 0..batch {
        let row = &data[b * classes..b * classes + classes];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        if pred == targets[b] {
            correct += 1;
        }
    }
    correct as f32 / batch as f32
}
