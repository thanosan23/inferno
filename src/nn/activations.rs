use super::{Module, Parameter};
use crate::tensor::Tensor;

macro_rules! activation_module {
    ($name:ident, $method:ident) => {
        pub struct $name;
        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.$method()
            }
            fn parameters(&self) -> Vec<Parameter> {
                vec![]
            }
        }
    };
}

activation_module!(ReLU, relu);
activation_module!(Sigmoid, sigmoid);
activation_module!(Tanh, tanh);
activation_module!(GELU, gelu);

pub struct Flatten;

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let n = shape[0];
        let rest: usize = shape[1..].iter().product();
        input.reshape(&[n, rest])
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
