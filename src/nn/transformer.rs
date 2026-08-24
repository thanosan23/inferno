use super::{LayerNorm, Linear, Module, Parameter};
use crate::tensor::Tensor;

pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub num_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, seed: u64) -> MultiHeadAttention {
        assert_eq!(d_model % num_heads, 0, "MultiHeadAttention: d_model {d_model} not divisible by num_heads {num_heads}");
        MultiHeadAttention {
            query: Linear::new(d_model, d_model, seed),
            key: Linear::new(d_model, d_model, seed + 1),
            value: Linear::new(d_model, d_model, seed + 2),
            output: Linear::new(d_model, d_model, seed + 3),
            num_heads,
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        let query = self.query.forward(input);
        let key = self.key.forward(input);
        let value = self.value.forward(input);
        let attended = Tensor::causal_self_attention(&query, &key, &value, self.num_heads);
        self.output.forward(&attended)
    }

    fn parameters(&self) -> Vec<Parameter> {
        [self.query.parameters(), self.key.parameters(), self.value.parameters(), self.output.parameters()].concat()
    }
}

pub struct FeedForward {
    pub up: Linear,
    pub down: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> FeedForward {
        FeedForward { up: Linear::new(d_model, d_ff, seed), down: Linear::new(d_ff, d_model, seed + 1) }
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.down.forward(&self.up.forward(input).gelu())
    }

    fn parameters(&self) -> Vec<Parameter> {
        [self.up.parameters(), self.down.parameters()].concat()
    }
}

pub struct TransformerBlock {
    pub pre_attention_norm: LayerNorm,
    pub attention: MultiHeadAttention,
    pub pre_feed_forward_norm: LayerNorm,
    pub feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, eps: f32, seed: u64) -> TransformerBlock {
        TransformerBlock {
            pre_attention_norm: LayerNorm::new(d_model, eps),
            attention: MultiHeadAttention::new(d_model, num_heads, seed),
            pre_feed_forward_norm: LayerNorm::new(d_model, eps),
            feed_forward: FeedForward::new(d_model, d_ff, seed + 10),
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let attended = self.attention.forward(&self.pre_attention_norm.forward(input));
        let after_attention = input.add(&attended);
        let fed_forward = self.feed_forward.forward(&self.pre_feed_forward_norm.forward(&after_attention));
        after_attention.add(&fed_forward)
    }

    fn parameters(&self) -> Vec<Parameter> {
        [
            self.pre_attention_norm.parameters(),
            self.attention.parameters(),
            self.pre_feed_forward_norm.parameters(),
            self.feed_forward.parameters(),
        ]
        .concat()
    }
}
