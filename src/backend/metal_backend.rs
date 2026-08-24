use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, DeviceRef, MTLResourceOptions, MTLSize,
};
use std::sync::OnceLock;

const SHADER_SOURCE: &str = include_str!("matmul.metal");

const MIN_ELEMENTS_FOR_GPU: usize = 128 * 128;

struct Context {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

static CONTEXT: OnceLock<Option<Context>> = OnceLock::new();

fn context() -> Option<&'static Context> {
    CONTEXT
        .get_or_init(|| {
            let device = Device::system_default()?;
            let library = device
                .new_library_with_source(SHADER_SOURCE, &metal::CompileOptions::new())
                .expect("failed to compile matmul.metal");
            let function = library
                .get_function("matmul", None)
                .expect("matmul kernel not found in compiled library");
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .expect("failed to build compute pipeline for matmul kernel");
            let queue = device.new_command_queue();
            Some(Context { queue, pipeline })
        })
        .as_ref()
}

pub fn is_available() -> bool {
    context().is_some()
}

#[repr(C)]
struct MatMulDims {
    m: u32,
    k: u32,
    n: u32,
}

fn buffer_from_slice(device: &DeviceRef, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub fn try_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if m * n < MIN_ELEMENTS_FOR_GPU {
        return None;
    }
    let ctx = context()?;
    let device = ctx.queue.device();

    objc::rc::autoreleasepool(|| {
        let buf_a = buffer_from_slice(device, a);
        let buf_b = buffer_from_slice(device, b);
        let out_len = m * n;
        let buf_out = device.new_buffer(
            (out_len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dims = MatMulDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
        };

        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MatMulDims>() as u64,
            &dims as *const MatMulDims as *const std::ffi::c_void,
        );

        let threads_per_group = MTLSize::new(16, 16, 1);
        let groups = MTLSize::new(
            (n as u64).div_ceil(16),
            (m as u64).div_ceil(16),
            1,
        );
        encoder.dispatch_thread_groups(groups, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_out.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
        Some(result)
    })
}
