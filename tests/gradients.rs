use inferno::nn;
use inferno::Tensor;

const FLOAT32_SAFE_EPSILON: f32 = 1e-2;

fn numerical_grad(x: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> Vec<f32> {
    let original = x.data();
    let mut grad = vec![0f32; original.len()];
    for i in 0..original.len() {
        let mut plus = original.clone();
        plus[i] += FLOAT32_SAFE_EPSILON;
        x.set_data(plus);
        let f_plus = f(x).item();

        let mut minus = original.clone();
        minus[i] -= FLOAT32_SAFE_EPSILON;
        x.set_data(minus);
        let f_minus = f(x).item();

        grad[i] = (f_plus - f_minus) / (2.0 * FLOAT32_SAFE_EPSILON);
    }
    x.set_data(original);
    grad
}

fn analytic_grad(x: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> Vec<f32> {
    x.set_requires_grad(true);
    x.zero_grad();
    let loss = f(x);
    loss.backward();
    x.grad().expect("backward() should have populated x's grad")
}

fn assert_close(name: &str, analytic: &[f32], numerical: &[f32]) {
    for (i, (a, n)) in analytic.iter().zip(numerical).enumerate() {
        let tol = 2e-2 + 2e-2 * n.abs();
        assert!(
            (a - n).abs() < tol,
            "{name}: grad mismatch at index {i}: analytic={a}, numerical={n}"
        );
    }
}

fn check(name: &str, x: &Tensor, f: impl Fn(&Tensor) -> Tensor) {
    let analytic = analytic_grad(x, &f);
    let numerical = numerical_grad(x, &f);
    assert_close(name, &analytic, &numerical);
}

#[test]
fn add_broadcast() {
    let x = Tensor::rand(&[4, 3], 1);
    let bias = Tensor::rand(&[3], 2);
    check("add_broadcast", &x, |x| x.add(&bias).sum());
}

#[test]
fn mul_and_sub() {
    let x = Tensor::rand(&[5], 3);
    let y = Tensor::rand(&[5], 4);
    check("mul_sub", &x, |x| x.mul(&y).sub(x).sum());
}

#[test]
fn matmul_grad() {
    let x = Tensor::rand(&[3, 4], 5);
    let w = Tensor::rand(&[4, 2], 6);
    check("matmul", &x, |x| x.matmul(&w).sum());
}

fn values_away_from_relu_kink(len: usize) -> Tensor {
    Tensor::from_fn(&[len], |i| 0.3 + i as f32 * 0.7 - 2.0)
}

#[test]
fn relu_and_sigmoid() {
    let x = values_away_from_relu_kink(6);
    check("relu_sigmoid", &x, |x| x.relu().sigmoid().sum());
}

#[test]
fn exp_ln_matmul_chain() {
    let x = Tensor::from_fn(&[3, 3], |i| 0.5 + i as f32 * 0.2);
    let w = Tensor::rand(&[3, 3], 7);
    check("exp_ln_matmul", &x, |x| x.exp().ln().matmul(&w).sum());
}

#[test]
fn log_softmax_grad() {
    let x = Tensor::rand(&[4, 5], 8);
    check("log_softmax", &x, |x| x.log_softmax().sum());
}

#[test]
fn conv2d_grad_through_smooth_activation() {
    let x = Tensor::rand(&[2, 3, 6, 6], 9);
    let w = Tensor::rand(&[4, 3, 3, 3], 10);
    check("conv2d", &x, |x| x.conv2d(&w, 1, 1).sigmoid().sum());
}

#[test]
fn conv2d_weight_grad() {
    let x = Tensor::rand(&[2, 3, 6, 6], 11);
    let w = Tensor::rand(&[4, 3, 3, 3], 12);
    check("conv2d_weight", &w, |w| x.conv2d(w, 1, 1).sum());
}

#[test]
fn max_pool2d_grad() {
    let x = Tensor::rand(&[2, 2, 6, 6], 13);
    check("max_pool2d", &x, |x| x.max_pool2d(2, 2).sum());
}

#[test]
fn avg_pool2d_grad() {
    let x = Tensor::rand(&[2, 2, 6, 6], 14);
    check("avg_pool2d", &x, |x| x.avg_pool2d(2, 2).sum());
}

#[test]
fn embedding_grad() {
    let table = Tensor::rand(&[10, 4], 15);
    let indices = [1usize, 3, 1, 7];
    check("embedding", &table, |t| t.embedding(&indices).sum());
}

#[test]
fn cross_entropy_grad() {
    let logits = Tensor::rand(&[4, 5], 16);
    let targets = [0usize, 2, 4, 1];
    check("cross_entropy", &logits, |l| nn::cross_entropy_loss(l, &targets));
}

#[test]
fn linear_layer_grad_flows_to_input() {
    use inferno::nn::{Linear, Module};
    let layer = Linear::new(4, 3, 42);
    let x = Tensor::rand(&[2, 4], 17);
    check("linear_layer", &x, |x| layer.forward(x).sum());
}

#[test]
fn linear_layer_grad_flows_through_rank_3_input() {
    use inferno::nn::{Linear, Module};
    let layer = Linear::new(4, 3, 43);
    let x = Tensor::rand(&[2, 5, 4], 18);
    check("linear_layer_rank3", &x, |x| layer.forward(x).sum());
}

#[test]
fn gelu_grad() {
    let x = Tensor::from_fn(&[8], |i| -2.0 + i as f32 * 0.6);
    check("gelu", &x, |x| x.gelu().sum());
}

#[test]
fn layer_norm_grad() {
    let x = Tensor::rand(&[4, 6], 19);
    check("layer_norm", &x, |x| x.layer_norm(1e-5).sum());
}

#[test]
fn layer_norm_module_grad() {
    use inferno::nn::{LayerNorm, Module};
    let layer = LayerNorm::new(6, 1e-5);
    let x = Tensor::rand(&[4, 6], 20);
    check("layer_norm_module", &x, |x| layer.forward(x).sum());
}

#[test]
fn causal_self_attention_grad_q() {
    let k = Tensor::rand(&[2, 5, 8], 21);
    let v = Tensor::rand(&[2, 5, 8], 22);
    let q = Tensor::rand(&[2, 5, 8], 23);
    check("attention_q", &q, |q| Tensor::causal_self_attention(q, &k, &v, 2).sum());
}

#[test]
fn causal_self_attention_grad_k() {
    let q = Tensor::rand(&[2, 5, 8], 24);
    let v = Tensor::rand(&[2, 5, 8], 25);
    let k = Tensor::rand(&[2, 5, 8], 26);
    check("attention_k", &k, |k| Tensor::causal_self_attention(&q, k, &v, 2).sum());
}

#[test]
fn causal_self_attention_grad_v() {
    let q = Tensor::rand(&[2, 5, 8], 27);
    let k = Tensor::rand(&[2, 5, 8], 28);
    let v = Tensor::rand(&[2, 5, 8], 29);
    check("attention_v", &v, |v| Tensor::causal_self_attention(&q, &k, v, 2).sum());
}

#[test]
fn causal_self_attention_ignores_future_tokens() {
    let num_heads = 2;
    let seq_len = 4;
    let head_dim = 8;
    let q = Tensor::rand(&[1, seq_len, head_dim], 30);
    let k = Tensor::rand(&[1, seq_len, head_dim], 31);
    let v = Tensor::rand(&[1, seq_len, head_dim], 32);
    let baseline_output = Tensor::causal_self_attention(&q, &k, &v, num_heads).data();

    let last_token_start = (seq_len - 1) * head_dim;
    let mut future_perturbed_v = v.data();
    for value in future_perturbed_v[last_token_start..last_token_start + head_dim].iter_mut() {
        *value += 5.0;
    }
    let perturbed_v = Tensor::new(future_perturbed_v, v.shape());
    let perturbed_output = Tensor::causal_self_attention(&q, &k, &perturbed_v, num_heads).data();

    let earlier_positions_len = last_token_start;
    assert_eq!(&baseline_output[..earlier_positions_len], &perturbed_output[..earlier_positions_len]);
}

#[test]
fn gpt_end_to_end_grad() {
    use inferno::nn::{GPTConfig, GPT};
    let vocab_size = 6;
    let (batch, seq_len) = (2, 4);
    let config = GPTConfig { vocab_size, block_size: seq_len, d_model: 8, num_heads: 2, num_layers: 2, d_ff: 16, eps: 1e-5 };
    let model = GPT::new(config, 42);
    let token_ids = [1usize, 2, 3, 4, 0, 5, 2, 1];
    let targets = [2usize, 3, 4, 0, 5, 2, 1, 3];

    check("gpt_token_embedding", &model.token_embedding.weight, |_| {
        let logits = model.forward(&token_ids, batch, seq_len).reshape(&[batch * seq_len, vocab_size]);
        nn::cross_entropy_loss(&logits, &targets)
    });
}
