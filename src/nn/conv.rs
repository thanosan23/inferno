use super::{Module, Parameter};
use crate::tensor::Tensor;

pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Parameter,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Conv2d {
        let fan_in = in_channels * kernel_size * kernel_size;
        let weight = Parameter::new(Tensor::kaiming_normal(
            &[out_channels, in_channels, kernel_size, kernel_size],
            fan_in,
            seed,
        ));
        let bias = Parameter::new(Tensor::zeros(&[out_channels]));
        Conv2d { weight, bias, stride, padding }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let out = input.conv2d(&self.weight, self.stride, self.padding);
        let out_channels = out.shape()[1];
        out.add(&self.bias.reshape(&[1, out_channels, 1, 1]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct Conv1d {
    pub weight: Parameter,
    pub bias: Parameter,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Conv1d {
        let fan_in = in_channels * kernel_size;
        let weight = Parameter::new(Tensor::kaiming_normal(
            &[out_channels, in_channels, kernel_size],
            fan_in,
            seed,
        ));
        let bias = Parameter::new(Tensor::zeros(&[out_channels]));
        Conv1d { weight, bias, stride, padding }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let out = input.conv1d(&self.weight, self.stride, self.padding);
        let out_channels = out.shape()[1];
        out.add(&self.bias.reshape(&[1, out_channels, 1]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct MaxPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel: usize, stride: usize) -> MaxPool2d {
        MaxPool2d { kernel, stride }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.kernel, self.stride)
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

pub struct AvgPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel: usize, stride: usize) -> AvgPool2d {
        AvgPool2d { kernel, stride }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.avg_pool2d(self.kernel, self.stride)
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
