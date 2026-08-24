use flate2::read::GzDecoder;
use inferno::backend;
use inferno::nn::{self, Conv2d, Flatten, Linear, MaxPool2d, Module, Sequential, ReLU};
use inferno::optim::{Adam, Optimizer, StepLR};
use inferno::Tensor;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Images {
    pixels: Vec<f32>,
    count: usize,
    rows: usize,
    cols: usize,
}

fn read_idx_bytes(path: &Path) -> Vec<u8> {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!("couldn't open {}: {e}\n\nrun `examples/download_mnist.sh` first to fetch the dataset.", path.display())
    });
    let mut bytes = Vec::new();
    if path.extension().is_some_and(|e| e == "gz") {
        GzDecoder::new(file).read_to_end(&mut bytes)
    } else {
        std::io::BufReader::new(file).read_to_end(&mut bytes)
    }
    .unwrap_or_else(|e| panic!("couldn't read {}: {e}", path.display()));
    bytes
}

fn be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn load_images(path: &Path) -> Images {
    let bytes = read_idx_bytes(path);
    assert_eq!(be_u32(&bytes, 0), 2051, "{}: not an MNIST image file", path.display());
    let count = be_u32(&bytes, 4) as usize;
    let rows = be_u32(&bytes, 8) as usize;
    let cols = be_u32(&bytes, 12) as usize;
    let pixels = bytes[16..16 + count * rows * cols].iter().map(|&b| b as f32 / 255.0).collect();
    Images { pixels, count, rows, cols }
}

fn load_labels(path: &Path) -> Vec<usize> {
    let bytes = read_idx_bytes(path);
    assert_eq!(be_u32(&bytes, 0), 2049, "{}: not an MNIST label file", path.display());
    let count = be_u32(&bytes, 4) as usize;
    bytes[8..8 + count].iter().map(|&b| b as usize).collect()
}

fn make_batch(images: &Images, labels: &[usize], indices: &[usize]) -> (Tensor, Vec<usize>) {
    let image_len = images.rows * images.cols;
    let mut batch_pixels = Vec::with_capacity(indices.len() * image_len);
    let mut batch_labels = Vec::with_capacity(indices.len());
    for &image_index in indices {
        batch_pixels.extend_from_slice(&images.pixels[image_index * image_len..(image_index + 1) * image_len]);
        batch_labels.push(labels[image_index]);
    }
    let batch_images = Tensor::new(batch_pixels, vec![indices.len(), 1, images.rows, images.cols]);
    (batch_images, batch_labels)
}

fn evaluate(model: &Sequential, images: &Images, labels: &[usize], batch_size: usize) -> f32 {
    let mut correct = 0usize;
    let mut start = 0;
    while start < images.count {
        let end = (start + batch_size).min(images.count);
        let indices: Vec<usize> = (start..end).collect();
        let (batch_images, batch_labels) = make_batch(images, labels, &indices);
        let logits = model.forward(&batch_images);
        correct += (nn::accuracy(&logits, &batch_labels) * indices.len() as f32).round() as usize;
        start = end;
    }
    correct as f32 / images.count as f32
}

fn main() {
    let data_dir: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data/mnist"));

    println!("matmul backend: {}", backend::active_backend());

    let train_images = load_images(&data_dir.join("train-images-idx3-ubyte.gz"));
    let train_labels = load_labels(&data_dir.join("train-labels-idx1-ubyte.gz"));
    let test_images = load_images(&data_dir.join("t10k-images-idx3-ubyte.gz"));
    let test_labels = load_labels(&data_dir.join("t10k-labels-idx1-ubyte.gz"));
    println!(
        "loaded {} training / {} test images ({}x{})",
        train_images.count, test_images.count, train_images.rows, train_images.cols
    );

    let model = Sequential::new(vec![
        Box::new(Conv2d::new(1, 8, 3, 1, 1, 0)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2)),
        Box::new(Conv2d::new(8, 16, 3, 1, 1, 1)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2)),
        Box::new(Flatten),
        Box::new(Linear::new(16 * 7 * 7, 128, 2)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, 3)),
    ]);
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("model: {param_count} trainable parameters");

    let mut optimizer = Adam::new(model.parameters(), 1e-3);
    let scheduler = StepLR::new(1e-3, 3, 0.5);

    let batch_size = 64;
    let epochs = 5;
    let n_batches = train_images.count / batch_size;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    for epoch in 0..epochs {
        optimizer.set_lr(scheduler.lr_at(epoch));
        let epoch_start = Instant::now();

        let mut order: Vec<usize> = (0..train_images.count).collect();
        order.shuffle(&mut rng);

        let mut total_loss = 0f32;
        for batch in 0..n_batches {
            let batch_indices = &order[batch * batch_size..(batch + 1) * batch_size];
            let (x, y) = make_batch(&train_images, &train_labels, batch_indices);

            let logits = model.forward(&x);
            let loss = nn::cross_entropy_loss(&logits, &y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item();
            if batch % 200 == 0 {
                println!("  epoch {epoch} batch {batch}/{n_batches} loss {:.4}", loss.item());
            }
        }

        let test_acc = evaluate(&model, &test_images, &test_labels, 500);
        println!(
            "epoch {epoch}: avg loss {:.4}, test accuracy {:.2}%, lr {:.5}, {:.1}s",
            total_loss / n_batches as f32,
            test_acc * 100.0,
            optimizer.lr(),
            epoch_start.elapsed().as_secs_f32(),
        );
    }
}
