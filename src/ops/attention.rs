use crate::backend;
use crate::ops::layout::transpose_matrix;
use crate::tensor::Tensor;

#[derive(Clone, Copy)]
struct AttentionGeometry {
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

impl AttentionGeometry {
    fn d_model(&self) -> usize {
        self.num_heads * self.head_dim
    }

    fn extract_head(&self, data: &[f32], batch_index: usize, head_index: usize) -> Vec<f32> {
        let d_model = self.d_model();
        let mut head = vec![0f32; self.seq_len * self.head_dim];
        for row in 0..self.seq_len {
            let src = (batch_index * self.seq_len + row) * d_model + head_index * self.head_dim;
            let dst = row * self.head_dim;
            head[dst..dst + self.head_dim].copy_from_slice(&data[src..src + self.head_dim]);
        }
        head
    }

    fn accumulate_head(&self, dest: &mut [f32], batch_index: usize, head_index: usize, head_grad: &[f32]) {
        let d_model = self.d_model();
        for row in 0..self.seq_len {
            let dst = (batch_index * self.seq_len + row) * d_model + head_index * self.head_dim;
            let src = row * self.head_dim;
            for c in 0..self.head_dim {
                dest[dst + c] += head_grad[src + c];
            }
        }
    }
}

fn causal_softmax_rows(scores: &mut [f32], seq_len: usize) {
    for row in 0..seq_len {
        let visible = &mut scores[row * seq_len..row * seq_len + seq_len];
        let max_visible = visible[..=row].iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0f32;
        for (col, value) in visible.iter_mut().enumerate() {
            *value = if col <= row {
                let weight = (*value - max_visible).exp();
                sum += weight;
                weight
            } else {
                0.0
            };
        }
        for value in visible[..=row].iter_mut() {
            *value /= sum;
        }
    }
}

fn softmax_backward_rows(weights: &[f32], weights_grad: &[f32], seq_len: usize) -> Vec<f32> {
    let mut scores_grad = vec![0f32; weights.len()];
    for row in 0..seq_len {
        let weights_row = &weights[row * seq_len..row * seq_len + seq_len];
        let weights_grad_row = &weights_grad[row * seq_len..row * seq_len + seq_len];
        let weighted_grad_sum: f32 = weights_row.iter().zip(weights_grad_row).map(|(&w, &g)| w * g).sum();
        for col in 0..seq_len {
            scores_grad[row * seq_len + col] = weights_row[col] * (weights_grad_row[col] - weighted_grad_sum);
        }
    }
    scores_grad
}

impl Tensor {
    pub fn causal_self_attention(q: &Tensor, k: &Tensor, v: &Tensor, num_heads: usize) -> Tensor {
        let shape = q.shape();
        assert_eq!(shape.len(), 3, "causal_self_attention: expected [batch, seq_len, d_model], got {:?}", shape);
        assert_eq!(k.shape(), shape, "causal_self_attention: q and k shapes must match");
        assert_eq!(v.shape(), shape, "causal_self_attention: q and v shapes must match");
        let (batch, seq_len, d_model) = (shape[0], shape[1], shape[2]);
        assert_eq!(d_model % num_heads, 0, "causal_self_attention: d_model {d_model} not divisible by num_heads {num_heads}");
        let head_dim = d_model / num_heads;
        let geometry = AttentionGeometry { seq_len, num_heads, head_dim };
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        let mut output = vec![0f32; batch * seq_len * d_model];
        let mut attention_weights = vec![0f32; batch * num_heads * seq_len * seq_len];

        for batch_index in 0..batch {
            for head_index in 0..num_heads {
                let q_head = geometry.extract_head(&q_data, batch_index, head_index);
                let k_head = geometry.extract_head(&k_data, batch_index, head_index);
                let v_head = geometry.extract_head(&v_data, batch_index, head_index);

                let k_head_transposed = transpose_matrix(&k_head, seq_len, head_dim);
                let mut scores = backend::matmul(&q_head, &k_head_transposed, seq_len, head_dim, seq_len);
                for score in scores.iter_mut() {
                    *score *= scale;
                }
                causal_softmax_rows(&mut scores, seq_len);

                let head_output = backend::matmul(&scores, &v_head, seq_len, seq_len, head_dim);
                geometry.accumulate_head(&mut output, batch_index, head_index, &head_output);

                let weights_offset = (batch_index * num_heads + head_index) * seq_len * seq_len;
                attention_weights[weights_offset..weights_offset + seq_len * seq_len].copy_from_slice(&scores);
            }
        }

        Tensor::make(output, vec![batch, seq_len, d_model], vec![q.clone(), k.clone(), v.clone()], move |output_grad| {
            let mut q_grad = vec![0f32; batch * seq_len * d_model];
            let mut k_grad = vec![0f32; batch * seq_len * d_model];
            let mut v_grad = vec![0f32; batch * seq_len * d_model];

            for batch_index in 0..batch {
                for head_index in 0..num_heads {
                    let q_head = geometry.extract_head(&q_data, batch_index, head_index);
                    let k_head = geometry.extract_head(&k_data, batch_index, head_index);
                    let v_head = geometry.extract_head(&v_data, batch_index, head_index);
                    let head_output_grad = geometry.extract_head(output_grad, batch_index, head_index);

                    let weights_offset = (batch_index * num_heads + head_index) * seq_len * seq_len;
                    let weights = &attention_weights[weights_offset..weights_offset + seq_len * seq_len];

                    let weights_transposed = transpose_matrix(weights, seq_len, seq_len);
                    let v_head_grad = backend::matmul(&weights_transposed, &head_output_grad, seq_len, seq_len, head_dim);

                    let v_head_transposed = transpose_matrix(&v_head, seq_len, head_dim);
                    let weights_grad = backend::matmul(&head_output_grad, &v_head_transposed, seq_len, head_dim, seq_len);

                    let mut scores_grad = softmax_backward_rows(weights, &weights_grad, seq_len);
                    for score_grad in scores_grad.iter_mut() {
                        *score_grad *= scale;
                    }

                    let q_head_grad = backend::matmul(&scores_grad, &k_head, seq_len, seq_len, head_dim);
                    let scores_grad_transposed = transpose_matrix(&scores_grad, seq_len, seq_len);
                    let k_head_grad = backend::matmul(&scores_grad_transposed, &q_head, seq_len, seq_len, head_dim);

                    geometry.accumulate_head(&mut q_grad, batch_index, head_index, &q_head_grad);
                    geometry.accumulate_head(&mut k_grad, batch_index, head_index, &k_head_grad);
                    geometry.accumulate_head(&mut v_grad, batch_index, head_index, &v_head_grad);
                }
            }

            vec![q_grad, k_grad, v_grad]
        })
    }
}
