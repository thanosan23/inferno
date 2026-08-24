use super::{Module, Parameter};
use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Parameter,
    pub bias: Parameter,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Linear {
        let weight = Parameter::new(Tensor::kaiming_normal(
            &[in_features, out_features],
            in_features,
            seed,
        ));
        let bias = Parameter::new(Tensor::zeros(&[out_features]));
        Linear { weight, bias, in_features, out_features }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weight).add(&self.bias)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
