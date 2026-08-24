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
        let shape = input.shape();
        let leading_dims = &shape[..shape.len() - 1];
        let rows: usize = leading_dims.iter().product();

        let flattened = input.reshape(&[rows, self.in_features]);
        let projected = flattened.matmul(&self.weight).add(&self.bias);

        let mut output_shape = leading_dims.to_vec();
        output_shape.push(self.out_features);
        projected.reshape(&output_shape)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
