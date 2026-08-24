use crate::tensor::Tensor;
use std::ops::Deref;

#[derive(Clone)]
pub struct Parameter(pub Tensor);

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        tensor.set_requires_grad(true);
        Parameter(tensor)
    }
}

impl Deref for Parameter {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.0
    }
}
