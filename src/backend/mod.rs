mod cpu;

#[cfg(feature = "metal-gpu")]
mod metal_backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "CPU"),
            Backend::Metal => write!(f, "Metal (GPU)"),
        }
    }
}

pub fn active_backend() -> Backend {
    #[cfg(feature = "metal-gpu")]
    {
        if metal_backend::is_available() {
            return Backend::Metal;
        }
    }
    Backend::Cpu
}

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "metal-gpu")]
    {
        if let Some(result) = metal_backend::try_matmul(a, b, m, k, n) {
            return result;
        }
    }
    cpu::matmul(a, b, m, k, n)
}
