use super::{Embedding, LayerNorm, Linear, Module, Parameter, TransformerBlock};
use crate::tensor::Tensor;

pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub eps: f32,
}

pub struct GPT {
    pub token_embedding: Embedding,
    pub position_embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub head: Linear,
    pub block_size: usize,
}

impl GPT {
    pub fn new(config: GPTConfig, seed: u64) -> GPT {
        let token_embedding = Embedding::new(config.vocab_size, config.d_model, seed);
        let position_embedding = Embedding::new(config.block_size, config.d_model, seed + 1);
        let blocks = (0..config.num_layers)
            .map(|layer| TransformerBlock::new(config.d_model, config.num_heads, config.d_ff, config.eps, seed + 100 + layer as u64 * 20))
            .collect();
        let final_norm = LayerNorm::new(config.d_model, config.eps);
        let head = Linear::new(config.d_model, config.vocab_size, seed + 2);

        GPT { token_embedding, position_embedding, blocks, final_norm, head, block_size: config.block_size }
    }

    pub fn forward(&self, token_ids: &[usize], batch: usize, seq_len: usize) -> Tensor {
        assert_eq!(token_ids.len(), batch * seq_len, "GPT::forward: expected {} token ids, got {}", batch * seq_len, token_ids.len());
        assert!(seq_len <= self.block_size, "GPT::forward: seq_len {seq_len} exceeds block_size {}", self.block_size);

        let d_model = self.token_embedding.weight.shape()[1];
        let token_embeddings = self.token_embedding.forward(token_ids).reshape(&[batch, seq_len, d_model]);

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let position_embeddings = self.position_embedding.forward(&position_ids);

        let mut hidden = token_embeddings.add(&position_embeddings);
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        let normalized = self.final_norm.forward(&hidden);
        self.head.forward(&normalized)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.token_embedding.parameters();
        params.extend(self.position_embedding.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.head.parameters());
        params
    }
}
