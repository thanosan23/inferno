# inferno

A small autograd + tensor + neural network framework, written from scratch
in Rust.

This was built on Apple Silicon, and so, naturally, inferno has support for the Metal GPU backend.

```
./examples/download_mnist.sh
cargo run --release --example mnist                      # CPU backend
cargo run --release --example mnist --features metal-gpu # GPU backend
```

There's also a GPT-style decoder-only transformer (`nn::GPT`: token + positional embeddings, causal self-attention, GELU feed-forward blocks, pre-norm residuals) trained on character-level Shakespeare:
```
./examples/download_shakespeare.sh
cargo run --release --example shakespeare
```