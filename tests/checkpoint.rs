use inferno::nn::{self, Parameter};
use inferno::Tensor;

fn temp_checkpoint_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("inferno_checkpoint_{name}.bin"))
}

#[test]
fn save_then_load_round_trips_parameter_values() {
    let path = temp_checkpoint_path("round_trip");
    let original = vec![
        Parameter::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])),
        Parameter::new(Tensor::new(vec![5.0, 6.0], vec![2])),
    ];

    nn::save(&original, &path).unwrap();

    let restored = vec![Parameter::new(Tensor::zeros(&[2, 2])), Parameter::new(Tensor::zeros(&[2]))];
    nn::load(&restored, &path).unwrap();

    for (expected, actual) in original.iter().zip(&restored) {
        assert_eq!(expected.data(), actual.data());
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
#[should_panic(expected = "shape")]
fn load_panics_on_shape_mismatch() {
    let path = temp_checkpoint_path("shape_mismatch");
    let original = vec![Parameter::new(Tensor::new(vec![1.0, 2.0], vec![2]))];
    nn::save(&original, &path).unwrap();

    let mismatched = vec![Parameter::new(Tensor::zeros(&[3]))];
    let _ = nn::load(&mismatched, &path);
}
