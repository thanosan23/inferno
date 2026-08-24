use crate::nn::Parameter;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&self);
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

pub struct SGD {
    params: Vec<Parameter>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(params: Vec<Parameter>, lr: f32) -> SGD {
        SGD::with_config(params, lr, 0.0, 0.0)
    }

    pub fn with_config(params: Vec<Parameter>, lr: f32, momentum: f32, weight_decay: f32) -> SGD {
        let velocity = params.iter().map(|p| vec![0f32; p.numel()]).collect();
        SGD { params, lr, momentum, weight_decay, velocity }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (p, v) in self.params.iter().zip(self.velocity.iter_mut()) {
            let Some(grad) = p.grad() else { continue };
            let mut data = p.data();
            for i in 0..data.len() {
                let g = grad[i] + self.weight_decay * data[i];
                v[i] = self.momentum * v[i] + g;
                data[i] -= self.lr * v[i];
            }
            p.set_data(data);
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

pub struct Adam {
    params: Vec<Parameter>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: i32,
}

impl Adam {
    pub fn new(params: Vec<Parameter>, lr: f32) -> Adam {
        Adam::with_config(params, lr, 0.9, 0.999, 1e-8, 0.0)
    }

    pub fn with_config(
        params: Vec<Parameter>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Adam {
        let m = params.iter().map(|p| vec![0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0f32; p.numel()]).collect();
        Adam { params, lr, beta1, beta2, eps, weight_decay, m, v, t: 0 }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t);

        for ((p, m), v) in self.params.iter().zip(self.m.iter_mut()).zip(self.v.iter_mut()) {
            let Some(grad) = p.grad() else { continue };
            let mut data = p.data();
            for i in 0..data.len() {
                if self.weight_decay != 0.0 {
                    data[i] -= self.lr * self.weight_decay * data[i];
                }
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;
                data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            p.set_data(data);
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

pub struct StepLR {
    pub base_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> StepLR {
        StepLR { base_lr, step_size, gamma }
    }

    pub fn lr_at(&self, epoch: usize) -> f32 {
        let decays = (epoch / self.step_size) as i32;
        self.base_lr * self.gamma.powi(decays)
    }
}
