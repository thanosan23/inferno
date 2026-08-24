# inferno

A small autograd + tensor + neural network framework, written from scratch
in Rust.

This was built on Apple Silicon, and so, naturally, inferno has support for the Metal GPU backend.

```
cargo run --release --example mnist                      # CPU backend
cargo run --release --example mnist --features metal-gpu # GPU backend
```