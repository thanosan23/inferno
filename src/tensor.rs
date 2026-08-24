use crate::autograd::{self, Op};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

pub(crate) struct TensorImpl {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,
    pub op: Option<Op>,
}

#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<RefCell<TensorImpl>>);

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data has {} elements but shape {:?} expects {}",
            data.len(),
            shape,
            expected
        );
        Tensor(Rc::new(RefCell::new(TensorImpl {
            data,
            shape,
            grad: None,
            requires_grad: false,
            op: None,
        })))
    }

    pub fn scalar(value: f32) -> Tensor {
        Tensor::new(vec![value], vec![1])
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::new(vec![0.0; shape.iter().product()], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::new(vec![1.0; shape.iter().product()], shape.to_vec())
    }

    pub fn rand(shape: &[usize], seed: u64) -> Tensor {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Uniform::new(-1.0f32, 1.0);
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| dist.sample(&mut rng)).collect();
        Tensor::new(data, shape.to_vec())
    }

    pub fn kaiming_normal(shape: &[usize], fan_in: usize, seed: u64) -> Tensor {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let std = (2.0 / fan_in as f32).sqrt();
        let dist = Normal::new(0.0f32, std).unwrap();
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| dist.sample(&mut rng)).collect();
        Tensor::new(data, shape.to_vec())
    }

    pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize, seed: u64) -> Tensor {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let dist = Uniform::new(-a, a);
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| dist.sample(&mut rng)).collect();
        Tensor::new(data, shape.to_vec())
    }

    pub fn from_fn(shape: &[usize], f: impl Fn(usize) -> f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::new((0..n).map(f).collect(), shape.to_vec())
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.0.borrow().data.len()
    }

    pub fn rank(&self) -> usize {
        self.0.borrow().shape.len()
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().data.clone()
    }

    pub fn item(&self) -> f32 {
        let inner = self.0.borrow();
        assert_eq!(inner.data.len(), 1, "item() requires a single-element tensor");
        inner.data[0]
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn set_requires_grad(&self, value: bool) {
        self.0.borrow_mut().requires_grad = value;
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn set_data(&self, data: Vec<f32>) {
        let mut inner = self.0.borrow_mut();
        assert_eq!(inner.data.len(), data.len());
        inner.data = data;
    }

    pub fn detach(&self) -> Tensor {
        let inner = self.0.borrow();
        Tensor::new(inner.data.clone(), inner.shape.clone())
    }

    pub fn backward(&self) {
        autograd::backward(self);
    }

    pub(crate) fn make(data: Vec<f32>, shape: Vec<usize>, parents: Vec<Tensor>, backward: impl Fn(&[f32]) -> Vec<Vec<f32>> + 'static) -> Tensor {
        let requires_grad = parents.iter().any(|p| p.requires_grad());
        let op = if requires_grad {
            Some(Op {
                parents,
                backward: Box::new(backward),
            })
        } else {
            None
        };
        Tensor(Rc::new(RefCell::new(TensorImpl {
            data,
            shape,
            grad: None,
            requires_grad,
            op,
        })))
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        f.debug_struct("Tensor")
            .field("shape", &inner.shape)
            .field("requires_grad", &inner.requires_grad)
            .field("data", &inner.data)
            .finish()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        write!(f, "Tensor(shape={:?}, {:?})", inner.shape, inner.data)
    }
}
