use super::{Module, Parameter};
use crate::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Parameter,
    pub beta: Parameter,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(feature_dim: usize, eps: f32) -> LayerNorm {
        LayerNorm {
            gamma: Parameter::new(Tensor::ones(&[feature_dim])),
            beta: Parameter::new(Tensor::zeros(&[feature_dim])),
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(self.eps).mul(&self.gamma).add(&self.beta)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
