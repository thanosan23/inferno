use crate::tensor::Tensor;

impl Tensor {
    pub fn layer_norm(&self, eps: f32) -> Tensor {
        let shape = self.shape();
        let feature_dim = *shape.last().expect("layer_norm: tensor must have at least one dimension");
        let rows = self.numel() / feature_dim;
        let data = self.data();

        let mut normalized = vec![0f32; data.len()];
        let mut inv_std_per_row = vec![0f32; rows];
        for row in 0..rows {
            let features = &data[row * feature_dim..row * feature_dim + feature_dim];
            let mean = features.iter().sum::<f32>() / feature_dim as f32;
            let variance = features.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / feature_dim as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();
            inv_std_per_row[row] = inv_std;
            for c in 0..feature_dim {
                normalized[row * feature_dim + c] = (features[c] - mean) * inv_std;
            }
        }

        let normalized_for_backward = normalized.clone();
        Tensor::make(normalized, shape, vec![self.clone()], move |grad_out| {
            let mut input_grad = vec![0f32; grad_out.len()];
            let n = feature_dim as f32;
            for row in 0..rows {
                let normalized_row = &normalized_for_backward[row * feature_dim..row * feature_dim + feature_dim];
                let grad_row = &grad_out[row * feature_dim..row * feature_dim + feature_dim];
                let inv_std = inv_std_per_row[row];
                let grad_sum: f32 = grad_row.iter().sum();
                let grad_dot_normalized: f32 = grad_row.iter().zip(normalized_row).map(|(&g, &y)| g * y).sum();
                for c in 0..feature_dim {
                    input_grad[row * feature_dim + c] =
                        inv_std / n * (n * grad_row[c] - grad_sum - normalized_row[c] * grad_dot_normalized);
                }
            }
            vec![input_grad]
        })
    }
}
