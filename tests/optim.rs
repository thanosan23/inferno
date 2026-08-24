use inferno::nn::Parameter;
use inferno::optim::{Adam, Optimizer, StepLR, SGD};
use inferno::Tensor;

fn param(start: f32) -> Parameter {
    Parameter::new(Tensor::new(vec![start], vec![1]))
}

fn step_towards_zero(p: &Parameter, optimizer: &mut impl Optimizer) {
    optimizer.zero_grad();
    p.mul(p).sum().backward();
    optimizer.step();
}

fn loss(p: &Parameter) -> f32 {
    p.mul(p).sum().item()
}

#[test]
fn sgd_step_reduces_loss() {
    let p = param(3.0);
    let mut optimizer = SGD::new(vec![p.clone()], 0.1);
    let loss_before = loss(&p);

    step_towards_zero(&p, &mut optimizer);

    assert!(loss(&p) < loss_before);
}

#[test]
fn adam_drives_loss_close_to_the_minimum() {
    let p = param(3.0);
    let mut optimizer = Adam::new(vec![p.clone()], 0.1);
    let loss_before = loss(&p);

    for _ in 0..50 {
        step_towards_zero(&p, &mut optimizer);
    }

    assert!(loss(&p) < loss_before * 0.01);
}

#[test]
fn param_without_gradient_is_untouched() {
    let touched = param(1.0);
    let untouched = param(5.0);
    let mut optimizer = SGD::new(vec![touched.clone(), untouched.clone()], 0.1);

    step_towards_zero(&touched, &mut optimizer);

    assert_eq!(untouched.data()[0], 5.0);
}

#[test]
fn zero_grad_clears_gradient() {
    let p = param(4.0);
    let optimizer = SGD::new(vec![p.clone()], 0.1);

    p.mul(&p).sum().backward();
    assert!(p.grad().is_some());

    optimizer.zero_grad();
    assert!(p.grad().is_none());
}

#[test]
fn set_lr_changes_step_size() {
    let mut optimizer = SGD::new(vec![param(1.0)], 0.1);
    optimizer.set_lr(1.0);
    assert_eq!(optimizer.lr(), 1.0);
}

#[test]
fn step_lr_decays_over_time() {
    let scheduler = StepLR::new(1.0, 3, 0.5);
    assert_eq!(scheduler.lr_at(0), 1.0);
    assert!(scheduler.lr_at(10) < scheduler.lr_at(0));
}
