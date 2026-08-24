use super::Parameter;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"INFC";
const VERSION: u32 = 1;

pub fn save(params: &[Parameter], path: impl AsRef<Path>) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&(params.len() as u32).to_le_bytes())?;

    for param in params {
        let shape = param.shape();
        writer.write_all(&(shape.len() as u32).to_le_bytes())?;
        for dim in shape {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }
        for value in param.data() {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()
}

pub fn load(params: &[Parameter], path: impl AsRef<Path>) -> io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    assert_eq!(&magic, MAGIC, "not an inferno checkpoint file");

    let version = read_u32(&mut reader)?;
    assert_eq!(version, VERSION, "checkpoint version {version} is not supported by this build (expected {VERSION})");

    let checkpoint_param_count = read_u32(&mut reader)? as usize;
    assert_eq!(
        checkpoint_param_count,
        params.len(),
        "checkpoint has {checkpoint_param_count} parameters but the model has {}",
        params.len()
    );

    for param in params {
        let rank = read_u32(&mut reader)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u32(&mut reader)? as usize);
        }
        assert_eq!(
            shape,
            param.shape(),
            "checkpoint parameter shape {:?} does not match model parameter shape {:?}",
            shape,
            param.shape()
        );

        let mut data = vec![0f32; shape.iter().product()];
        for value in data.iter_mut() {
            *value = read_f32(&mut reader)?;
        }
        param.set_data(data);
    }
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
