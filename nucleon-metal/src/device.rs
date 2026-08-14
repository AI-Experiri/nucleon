//! Metal plumbing: device, queue, kernel compilation, buffers, dispatch.
//!
//! One `Gpu` per process. It compiles `kernels/ops.metal` at startup into a
//! pipeline per kernel function, and exposes the small set of primitives
//! every op wrapper needs: make a buffer, encode a dispatch, wait, read.
//!
//! The `unsafe` blocks are the Rust/GPU memory boundary: copying bytes into
//! a shared buffer and reading results back. On Apple Silicon
//! `StorageModeShared` buffers are ordinary RAM visible to both CPU and
//! GPU, so there are no transfers — only visibility rules (never read while
//! the GPU may still be writing; `run` waits for completion).

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLibrary, MTLResource, MTLResourceOptions, MTLSize, MTLStorageMode,
};

// MTLCreateSystemDefaultDevice needs CoreGraphics linked or it returns nil.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Every kernel function in kernels/ops.metal. Compiled once at startup.
const KERNEL_NAMES: [&str; 12] = [
    "add",
    "mul",
    "silu",
    "embed",
    "matvec",
    "matmul",
    "rope",
    "rmsnorm",
    "softmax",
    "argmax",
    "rmsnorm_matvec",
    "attention",
];

/// Threadgroup size for the reduction kernels. Must match the scratch array
/// size in ops.metal and be a power of two (the tree reduction halves it).
pub const REDUCTION_THREADS: usize = 256;

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Gpu {
    device: Device,
    queue: Queue,
    pipelines: HashMap<&'static str, Pipeline>,
}

impl Gpu {
    /// Set up the GPU: device, queue, compile all kernels. Returns a
    /// descriptive error when there is no GPU (headless CI) or a kernel
    /// fails to compile (the NSError text is the compiler diagnostic —
    /// swallowing it makes shader errors invisible).
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        // dispatchThreads (nonuniform threadgroups) is only defined on
        // Apple4+ GPUs — every Apple Silicon chip. Older/foreign GPUs
        // (e.g. Intel-Mac eGPUs) would hit undefined dispatch behavior.
        if !device.supportsFamily(MTLGPUFamily::Apple4) {
            return Err("GPU lacks nonuniform threadgroup support (Apple Silicon required)".into());
        }
        let queue = device
            .newCommandQueue()
            .ok_or("failed to create command queue")?;

        let source = NSString::from_str(include_str!("../kernels/ops.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| format!("kernel compile failed: {}", e.localizedDescription()))?;

        let mut pipelines = HashMap::new();
        for name in KERNEL_NAMES {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("kernel function '{name}' not found in ops.metal"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("pipeline for '{name}': {}", e.localizedDescription()))?;
            pipelines.insert(name, pipeline);
        }
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    /// A shared-memory buffer initialized with `data`.
    pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
        self.buffer_from_bytes(data.as_ptr().cast(), std::mem::size_of_val(data))
    }

    pub fn buffer_from_u32(&self, data: &[u32]) -> Buffer {
        self.buffer_from_bytes(data.as_ptr().cast(), std::mem::size_of_val(data))
    }

    /// A shared-memory buffer for `len` f32 results. Metal documents
    /// newBufferWithLength as a ZERO-FILLED allocation, so reading a buffer
    /// no kernel wrote is defined (zeros), never uninitialized memory —
    /// the safe read_f32/read_u32 APIs depend on this guarantee.
    pub fn buffer_for_output(&self, len: usize) -> Buffer {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .expect("output byte count overflows usize");
        self.device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .expect("buffer allocation failed")
    }

    fn buffer_from_bytes(&self, ptr: *const std::ffi::c_void, bytes: usize) -> Buffer {
        if bytes == 0 {
            // an empty slice's pointer is dangling; never ask Metal to copy
            // from it — hand back a fresh (min-size) buffer instead
            return self.buffer_for_output(0);
        }
        let ptr = NonNull::new(ptr.cast_mut()).expect("null data pointer");
        unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer allocation failed")
        }
    }

    /// Copy a buffer's contents back into a Vec. Only call after `run`
    /// returned — reading while the GPU writes is a data race. Panics if
    /// `len` elements exceed the buffer's actual allocation (reading past
    /// it from safe code would be undefined behavior). Buffers must come
    /// from this `Gpu` (StorageModeShared, zero-filled at creation) — a
    /// foreign private-storage MTLBuffer's contents() is not CPU-readable.
    pub fn read_f32(&self, buffer: &Buffer, len: usize) -> Vec<f32> {
        self.check_read_bounds(buffer, len, std::mem::size_of::<f32>());
        let ptr = buffer.contents().as_ptr().cast::<f32>();
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    pub fn read_u32(&self, buffer: &Buffer, len: usize) -> Vec<u32> {
        self.check_read_bounds(buffer, len, std::mem::size_of::<u32>());
        let ptr = buffer.contents().as_ptr().cast::<u32>();
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn check_read_bounds(&self, buffer: &Buffer, len: usize, elem_size: usize) {
        // a private-storage buffer's contents() is NULL (not CPU-mapped);
        // objc2 models it NonNull, so dereferencing would be UB from safe
        // code — reject anything that is not shared storage first
        assert_eq!(
            buffer.storageMode(),
            MTLStorageMode::Shared,
            "read requires a StorageModeShared buffer from this Gpu"
        );
        // checked: a wrapping len * elem_size in release mode would pass a
        // small byte count here while from_raw_parts uses the huge len
        let bytes = len
            .checked_mul(elem_size)
            .expect("read byte count overflows usize");
        let capacity = buffer.length();
        assert!(
            bytes <= capacity,
            "read of {bytes} bytes exceeds buffer capacity {capacity}"
        );
    }

    /// Encode one kernel launch and block until the GPU finishes.
    ///
    /// `buffers` bind in order to [[buffer(0..)]]; `scalars` (raw little
    /// blobs for `constant uint&` / `constant float&` params) bind after
    /// them. `grid` is the total thread count in x/y; `threadgroup` is the
    /// per-group size (map ops pass None and get a derived size).
    ///
    /// # Safety
    /// The caller must uphold the target kernel's contract: buffers large
    /// enough for every index the kernel computes, correct scalar count and
    /// types, and a grid/threadgroup shape the kernel was written for.
    /// Additionally: all buffers must come from THIS device; no other CPU
    /// or GPU work may mutate them until this call returns (it waits for
    /// completion before returning, which is what makes later reads safe);
    /// and a buffer bound as kernel output must not alias a buffer the
    /// same kernel reads unless the kernel is written for it. Buffers are
    /// shared process memory — a kernel writing out of bounds corrupts
    /// arbitrary memory, which is undefined behavior.
    pub unsafe fn run(
        &self,
        kernel: &str,
        buffers: &[&Buffer],
        scalars: &[Scalar],
        grid: (usize, usize),
        threadgroup: Option<(usize, usize)>,
    ) {
        let pipeline = &self.pipelines[kernel];
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
        }
        for (j, scalar) in scalars.iter().enumerate() {
            let (ptr, len) = scalar.raw();
            unsafe { enc.setBytes_length_atIndex(ptr, len, buffers.len() + j) };
        }

        let (tg_w, tg_h) = threadgroup.unwrap_or_else(|| {
            // map ops: a flat or 2-D grid; cap by what the pipeline allows
            let max = pipeline.maxTotalThreadsPerThreadgroup();
            if grid.1 > 1 {
                (16, 16.min(max / 16).max(1))
            } else {
                (max.min(256), 1)
            }
        });
        let grid_size = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: tg_w,
            height: tg_h,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        check_completed(&cmd);
    }
}

/// A command buffer that finished in any state other than Completed (shader
/// fault, validation failure, OOB access) must not be treated as success —
/// its output buffers hold stale or garbage data. Panic with the GPU error.
fn check_completed(cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
    let status = cmd.status();
    if status != MTLCommandBufferStatus::Completed {
        let detail = cmd
            .error()
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| "no error detail".into());
        panic!("GPU command buffer failed (status {status:?}): {detail}");
    }
}

/// One dispatch inside a batched command buffer (see [`Gpu::run_many`]).
pub struct KernelCall<'a> {
    pub kernel: &'a str,
    pub buffers: Vec<&'a Buffer>,
    pub scalars: Vec<Scalar>,
    pub grid: (usize, usize),
    pub threadgroup: Option<(usize, usize)>,
}

impl Gpu {
    /// Encode MANY dispatches into ONE command buffer, commit once, wait
    /// once. This is how a real engine submits a forward pass; the per-call
    /// `run` above pays a commit + wait per op. Comparing the two (and
    /// comparing dispatch counts within this one) is the fusion lesson.
    ///
    /// # Safety
    /// Same contract as [`Gpu::run`] for every call, with one refinement:
    /// dispatches inside ONE command buffer execute in encode order, so a
    /// later call may read buffers an earlier call wrote (the fusion
    /// benchmark relies on this). The no-external-mutation rule applies to
    /// the whole batch duration instead of per call.
    pub unsafe fn run_many(&self, calls: &[KernelCall<'_>]) {
        let cmd = self.queue.commandBuffer().expect("command buffer");
        for call in calls {
            let pipeline = &self.pipelines[call.kernel];
            let enc = cmd.computeCommandEncoder().expect("compute encoder");
            enc.setComputePipelineState(pipeline);
            for (i, buf) in call.buffers.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
            }
            for (j, scalar) in call.scalars.iter().enumerate() {
                let (ptr, len) = scalar.raw();
                unsafe { enc.setBytes_length_atIndex(ptr, len, call.buffers.len() + j) };
            }
            let (tg_w, tg_h) = call.threadgroup.unwrap_or_else(|| {
                let max = pipeline.maxTotalThreadsPerThreadgroup();
                if call.grid.1 > 1 {
                    (16, 16.min(max / 16).max(1))
                } else {
                    (max.min(256), 1)
                }
            });
            let grid_size = MTLSize {
                width: call.grid.0,
                height: call.grid.1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        check_completed(&cmd);
    }
}

/// A scalar kernel argument, passed by value via setBytes.
pub enum Scalar {
    U32(u32),
    F32(f32),
}

impl Scalar {
    fn raw(&self) -> (NonNull<std::ffi::c_void>, usize) {
        match self {
            Scalar::U32(v) => (NonNull::from(v).cast(), std::mem::size_of::<u32>()),
            Scalar::F32(v) => (NonNull::from(v).cast(), std::mem::size_of::<f32>()),
        }
    }
}

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
