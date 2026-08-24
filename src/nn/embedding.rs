use super::{Module, Parameter};
use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: Parameter,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, seed: u64) -> Embedding {
        let weight = Parameter::new(Tensor::xavier_uniform(
            &[num_embeddings, embedding_dim],
            num_embeddings,
            embedding_dim,
            seed,
        ));
        Embedding { weight }
    }

    pub fn forward(&self, indices: &[usize]) -> Tensor {
        self.weight.embedding(indices)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let indices: Vec<usize> = input.data().iter().map(|&x| x as usize).collect();
        self.weight.embedding(&indices)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}
