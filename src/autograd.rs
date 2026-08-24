use crate::tensor::Tensor;
use std::collections::HashSet;
use std::rc::Rc;

pub(crate) type BackwardFn = Box<dyn Fn(&[f32]) -> Vec<Vec<f32>>>;

pub(crate) struct Op {
    pub parents: Vec<Tensor>,
    pub backward: BackwardFn,
}

fn tensor_id(t: &Tensor) -> usize {
    Rc::as_ptr(&t.0) as usize
}

fn topo_order(root: &Tensor) -> Vec<Tensor> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Tensor, usize)> = vec![(root.clone(), 0)];
    while let Some((node, child_idx)) = stack.pop() {
        let id = tensor_id(&node);
        if child_idx == 0 {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
        }
        let n_parents = node.0.borrow().op.as_ref().map_or(0, |op| op.parents.len());
        if child_idx < n_parents {
            let next_parent = node.0.borrow().op.as_ref().unwrap().parents[child_idx].clone();
            stack.push((node.clone(), child_idx + 1));
            if !visited.contains(&tensor_id(&next_parent)) {
                stack.push((next_parent, 0));
            }
        } else {
            order.push(node);
        }
    }
    order
}

pub(crate) fn backward(root: &Tensor) {
    assert_eq!(
        root.numel(),
        1,
        "backward() can only be called on a scalar tensor (got shape {:?}); \
         reduce it with .sum() or .mean() first",
        root.shape()
    );
    root.0.borrow_mut().grad = Some(vec![1.0]);

    let order = topo_order(root);
    for node in order.into_iter().rev() {
        let grad = node.0.borrow().grad.clone();
        let Some(grad) = grad else { continue };
        let op = node.0.borrow_mut().op.take();
        let Some(op) = op else { continue };
        let parent_grads = (op.backward)(&grad);
        for (parent, g) in op.parents.iter().zip(parent_grads) {
            if !parent.requires_grad() {
                continue;
            }
            let mut inner = parent.0.borrow_mut();
            match &mut inner.grad {
                Some(existing) => {
                    for (e, gi) in existing.iter_mut().zip(g.iter()) {
                        *e += gi;
                    }
                }
                None => inner.grad = Some(g),
            }
        }
        node.0.borrow_mut().op = Some(op);
    }
}
