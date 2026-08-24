use super::Parameter;
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn parameters(&self) -> Vec<Parameter>;
}
