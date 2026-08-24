use inferno::backend;
use inferno::nn::{self, GPTConfig, GPT};
use inferno::optim::{Adam, Optimizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

struct CharVocab {
    id_to_char: Vec<char>,
    char_to_id: std::collections::HashMap<char, usize>,
}

impl CharVocab {
    fn build(text: &str) -> CharVocab {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort_unstable();
        let char_to_id = chars.iter().enumerate().map(|(id, &c)| (c, id)).collect();
        CharVocab { id_to_char: chars, char_to_id }
    }

    fn len(&self) -> usize {
        self.id_to_char.len()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.char_to_id[&c]).collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.id_to_char[id]).collect()
    }
}

fn sample_batch(data: &[usize], block_size: usize, batch_size: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(batch_size * block_size);
    let mut targets = Vec::with_capacity(batch_size * block_size);
    for _ in 0..batch_size {
        let start = rng.gen_range(0..data.len() - block_size - 1);
        inputs.extend_from_slice(&data[start..start + block_size]);
        targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
    }
    (inputs, targets)
}

fn estimate_loss(model: &GPT, data: &[usize], block_size: usize, batch_size: usize, vocab_size: usize, rng: &mut StdRng, iters: usize) -> f32 {
    let mut total_loss = 0f32;
    for _ in 0..iters {
        let (inputs, targets) = sample_batch(data, block_size, batch_size, rng);
        let logits = model.forward(&inputs, batch_size, block_size).reshape(&[batch_size * block_size, vocab_size]);
        total_loss += nn::cross_entropy_loss(&logits, &targets).item();
    }
    total_loss / iters as f32
}

fn sample_next_token(logits: &[f32], rng: &mut StdRng) -> usize {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let total_weight: f32 = weights.iter().sum();
    let target = rng.gen::<f32>() * total_weight;

    let mut cumulative = 0f32;
    for (id, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= target {
            return id;
        }
    }
    weights.len() - 1
}

fn generate(model: &GPT, vocab: &CharVocab, prompt: &[usize], num_new_tokens: usize, block_size: usize, rng: &mut StdRng) -> String {
    let mut context = prompt.to_vec();
    let mut generated_ids = Vec::with_capacity(num_new_tokens);
    for _ in 0..num_new_tokens {
        let window_start = context.len().saturating_sub(block_size);
        let window = &context[window_start..];
        let seq_len = window.len();

        let logits = model.forward(window, 1, seq_len).reshape(&[seq_len, vocab.len()]);
        let logits_data = logits.data();
        let last_token_logits = &logits_data[(seq_len - 1) * vocab.len()..seq_len * vocab.len()];

        let next_id = sample_next_token(last_token_logits, rng);
        context.push(next_id);
        generated_ids.push(next_id);
    }
    vocab.decode(&generated_ids)
}

fn main() {
    let corpus_path: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data/shakespeare/input.txt"));
    let text = std::fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
        panic!("couldn't read {}: {e}\n\nrun `examples/download_shakespeare.sh` first to fetch the corpus.", corpus_path.display())
    });

    println!("matmul backend: {}", backend::active_backend());

    let vocab = CharVocab::build(&text);
    let encoded = vocab.encode(&text);
    let split_at = encoded.len() * 9 / 10;
    let (train_data, val_data) = encoded.split_at(split_at);
    println!("corpus: {} characters, {} unique, {} train / {} val", encoded.len(), vocab.len(), train_data.len(), val_data.len());

    let block_size = 32;
    let config = GPTConfig { vocab_size: vocab.len(), block_size, d_model: 64, num_heads: 4, num_layers: 2, d_ff: 192, eps: 1e-5 };
    let model = GPT::new(config, 1);
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("model: {param_count} trainable parameters");

    let batch_size = 32;
    let num_steps = 2000;
    let mut optimizer = Adam::new(model.parameters(), 3e-3);
    let mut rng = StdRng::seed_from_u64(0);

    for step in 0..num_steps {
        let (inputs, targets) = sample_batch(train_data, block_size, batch_size, &mut rng);
        let logits = model.forward(&inputs, batch_size, block_size).reshape(&[batch_size * block_size, vocab.len()]);
        let loss = nn::cross_entropy_loss(&logits, &targets);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if step % 200 == 0 || step == num_steps - 1 {
            let val_loss = estimate_loss(&model, val_data, block_size, batch_size, vocab.len(), &mut rng, 20);
            println!("step {step}: train loss {:.4}, val loss {:.4}", loss.item(), val_loss);
        }
    }

    let prompt = vocab.encode("\n");
    let sample = generate(&model, &vocab, &prompt, 500, block_size, &mut rng);
    println!("\n--- generated ---\n{sample}");
}
