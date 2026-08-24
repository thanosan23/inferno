mod attention;
mod broadcast;
mod conv;
mod layout;
mod normalization;

use crate::backend;
use crate::ops::broadcast::{broadcast_binary, broadcast_shape, reduce_to_shape};
use crate::ops::layout::transpose_matrix;
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn binary_op(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
    grad_a: impl Fn(f32, f32) -> f32 + 'static,
    grad_b: impl Fn(f32, f32) -> f32 + 'static,
) -> Tensor {
    let a_data = a.data();
    let a_shape = a.shape();
    let b_data = b.data();
    let b_shape = b.shape();
    let out_shape = broadcast_shape(&a_shape, &b_shape);
    let out_data = broadcast_binary(&a_data, &a_shape, &b_data, &b_shape, &out_shape, &f);

    let backward_shape = out_shape.clone();
    Tensor::make(out_data, out_shape, vec![a.clone(), b.clone()], move |grad_out| {
        let a_local_grad = broadcast_binary(&a_data, &a_shape, &b_data, &b_shape, &backward_shape, &grad_a);
        let a_grad_full: Vec<f32> = grad_out.iter().zip(&a_local_grad).map(|(g, l)| g * l).collect();

        let b_local_grad = broadcast_binary(&a_data, &a_shape, &b_data, &b_shape, &backward_shape, &grad_b);
        let b_grad_full: Vec<f32> = grad_out.iter().zip(&b_local_grad).map(|(g, l)| g * l).collect();

        vec![
            reduce_to_shape(&a_grad_full, &backward_shape, &a_shape),
            reduce_to_shape(&b_grad_full, &backward_shape, &b_shape),
        ]
    })
}

fn unary_op(a: &Tensor, f: impl Fn(f32) -> f32, grad: impl Fn(f32) -> f32 + 'static) -> Tensor {
    let input_data = a.data();
    let shape = a.shape();
    let out_data: Vec<f32> = input_data.iter().map(|&x| f(x)).collect();
    Tensor::make(out_data, shape, vec![a.clone()], move |grad_out| {
        vec![grad_out.iter().zip(&input_data).map(|(g, &x)| g * grad(x)).collect()]
    })
}

const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_CUBIC_COEFFICIENT: f32 = 0.044715;

fn gelu_tanh_argument(x: f32) -> f32 {
    GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x * x * x)
}

fn gelu_value(x: f32) -> f32 {
    0.5 * x * (1.0 + gelu_tanh_argument(x).tanh())
}

fn gelu_derivative(x: f32) -> f32 {
    let tanh_term = gelu_tanh_argument(x).tanh();
    let tanh_argument_derivative = GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC_COEFFICIENT * x * x);
    0.5 * (1.0 + tanh_term) + 0.5 * x * (1.0 - tanh_term * tanh_term) * tanh_argument_derivative
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |a, b| a + b, |_, _| 1.0, |_, _| 1.0)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |a, b| a - b, |_, _| 1.0, |_, _| -1.0)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |a, b| a * b, |_, b| b, |a, _| a)
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        binary_op(self, other, |a, b| a / b, |_, b| 1.0 / b, |a, b| -a / (b * b))
    }

    pub fn neg(&self) -> Tensor {
        unary_op(self, |x| -x, |_| -1.0)
    }

    pub fn relu(&self) -> Tensor {
        unary_op(self, |x| x.max(0.0), |x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    pub fn sigmoid(&self) -> Tensor {
        unary_op(
            self,
            |x| 1.0 / (1.0 + (-x).exp()),
            |x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            },
        )
    }

    pub fn tanh(&self) -> Tensor {
        unary_op(self, |x| x.tanh(), |x| 1.0 - x.tanh() * x.tanh())
    }

    pub fn gelu(&self) -> Tensor {
        unary_op(self, gelu_value, gelu_derivative)
    }

    pub fn exp(&self) -> Tensor {
        unary_op(self, |x| x.exp(), |x| x.exp())
    }

    pub fn ln(&self) -> Tensor {
        unary_op(self, |x| x.ln(), |x| 1.0 / x)
    }

    pub fn powf(&self, p: f32) -> Tensor {
        unary_op(self, move |x| x.powf(p), move |x| p * x.powf(p - 1.0))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a_shape = self.shape();
        let b_shape = other.shape();
        assert_eq!(a_shape.len(), 2, "matmul: lhs must be 2D, got {:?}", a_shape);
        assert_eq!(b_shape.len(), 2, "matmul: rhs must be 2D, got {:?}", b_shape);
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        assert_eq!(k, k2, "matmul: inner dims must match, got {:?} and {:?}", a_shape, b_shape);

        let a_data = self.data();
        let b_data = other.data();
        let out_data = backend::matmul(&a_data, &b_data, m, k, n);

        Tensor::make(out_data, vec![m, n], vec![self.clone(), other.clone()], move |grad_out| {
            let b_transposed = transpose_matrix(&b_data, k, n);
            let a_grad = backend::matmul(grad_out, &b_transposed, m, n, k);

            let a_transposed = transpose_matrix(&a_data, m, k);
            let b_grad = backend::matmul(&a_transposed, grad_out, k, m, n);

            vec![a_grad, b_grad]
        })
    }

    pub fn t(&self) -> Tensor {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "t(): expected a 2D tensor, got {:?}", shape);
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.data();
        let transposed = transpose_matrix(&data, rows, cols);
        Tensor::make(transposed, vec![cols, rows], vec![self.clone()], move |grad_out| {
            vec![transpose_matrix(grad_out, cols, rows)]
        })
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: {:?} does not match {} elements", shape, self.numel());
        let data = self.data();
        let shape = shape.to_vec();
        Tensor::make(data, shape, vec![self.clone()], move |grad_out| vec![grad_out.to_vec()])
    }

    pub fn sum(&self) -> Tensor {
        let data = self.data();
        let n = data.len();
        let total: f32 = data.iter().sum();
        Tensor::make(vec![total], vec![1], vec![self.clone()], move |grad_out| {
            vec![vec![grad_out[0]; n]]
        })
    }

    pub fn mean(&self) -> Tensor {
        let n = self.numel() as f32;
        self.sum().mul(&Tensor::scalar(1.0 / n))
    }

    pub fn log_softmax(&self) -> Tensor {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "log_softmax: expected a 2D tensor, got {:?}", shape);
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.data();

        let mut log_probs = vec![0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * cols..r * cols + cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            let log_sum_exp = sum_exp.ln() + max;
            for c in 0..cols {
                log_probs[r * cols + c] = row[c] - log_sum_exp;
            }
        }

        let log_probs_for_backward = log_probs.clone();
        Tensor::make(log_probs, vec![rows, cols], vec![self.clone()], move |grad_out| {
            let mut input_grad = vec![0f32; log_probs_for_backward.len()];
            for r in 0..rows {
                let row_log_probs = &log_probs_for_backward[r * cols..r * cols + cols];
                let row_grad_out = &grad_out[r * cols..r * cols + cols];
                let grad_sum: f32 = row_grad_out.iter().sum();
                for c in 0..cols {
                    let softmax_c = row_log_probs[c].exp();
                    input_grad[r * cols + c] = row_grad_out[c] - softmax_c * grad_sum;
                }
            }
            vec![input_grad]
        })
    }
}

macro_rules! impl_op_trait {
    ($trait_:ident, $method:ident, $tensor_method:ident) => {
        impl $trait_ for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$tensor_method(self, rhs)
            }
        }
        impl $trait_ for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$tensor_method(&self, &rhs)
            }
        }
        impl $trait_<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$tensor_method(&self, rhs)
            }
        }
        impl $trait_<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$tensor_method(self, &rhs)
            }
        }
    };
}

impl_op_trait!(Add, add, add);
impl_op_trait!(Sub, sub, sub);
impl_op_trait!(Mul, mul, mul);
impl_op_trait!(Div, div, div);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
