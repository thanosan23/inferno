use super::{Module, Parameter};
use crate::tensor::Tensor;

pub struct Sequential(pub Vec<Box<dyn Module>>);

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential(layers)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.0 {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.0.iter().flat_map(|layer| layer.parameters()).collect()
    }
}
