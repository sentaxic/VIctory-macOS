//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

mod flight;
pub(crate) mod same_copy;
pub(crate) use flight::{async_submit_enabled, flight1_enabled, probe_marked};
#[cfg(test)]
pub(crate) use flight::{Gate, ModeGuard};

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::backend::{HostGpu, HostMemory};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// A console state as a host texture state: PC bits only (not the console's
/// own, compression hints from bit 25 up, which the host rejects), and none
/// of the buffer-only ones the console allows on textures (vertex, constant
/// and index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    state & 0x01ff_ffff & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

thread_local! {
    /// The thread holds the submission lock ([`HostGpu::execute`] or
    /// [`HostGpu::quiet`]).
    static HOLDS_SUBMISSIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The thread holds the submission lock while this lives.
struct Holding(bool);

impl Holding {
    fn enter() -> Self {
        Holding(HOLDS_SUBMISSIONS.with(|h| h.replace(true)))
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        HOLDS_SUBMISSIONS.with(|h| h.set(self.0));
    }
}

/// Submissions held off ([`HostGpu::quiet`]).
pub(crate) struct Quiet<'a> {
    holding: Option<Holding>,
    held: Option<std::sync::MutexGuard<'a, u64>>,
    landed: bool,
}

impl Drop for Quiet<'_> {
    fn drop(&mut self) {
        // Drop in strict order: thread-local flag, then submission lock, and
        // only after releasing `submitted` notify held queues if a flight landed.
        self.holding.take();
        self.held.take();
        if self.landed {
            crate::queue::on_flight_landed();
        }
    }
}

/// Prepared submission artifacts owned between `prepare_submission` and
/// `finish_post_gpu`.
struct PreparedSubmission {
    lists: Vec<Option<ID3D12CommandList>>,
    #[cfg(feature = "diagnostics")]
    firsts: Vec<usize>,
    native_boundaries: Vec<usize>,
    batch: Batch,
    heaps: Vec<std::sync::Arc<crate::bindless::Viewed>>,
    uploads: Option<crate::bindless::Uploads>,
    started: std::time::Instant,
    prepared: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    traced: bool,
}

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    /// The direct queue; a new one replaces it when it stops running work
    /// ([`Host::lost`]).
    queue: Mutex<ID3D12CommandQueue>,
    pub(crate) depth_bounds_supported: bool,
    enhanced_barriers_supported: bool,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// The list every submission ends with, made on first use ([`Mark`]).
    mark: OnceLock<Option<Mark>>,
    /// With `VICTORY_PREP_AHEAD=1`, two host submissions may be in flight:
    /// those with an even fence value end with this mark instead
    /// ([`Host::mark_for`]), so arming one never changes the other's.
    mark_even: OnceLock<Option<Mark>>,
    /// Extra marks for native calls before the final call of a split submission.
    split_marks: Mutex<Option<Vec<Mark>>>,
    /// Reusable global UAV barrier lists inserted between coalesced queue pieces.
    barrier_lists: Mutex<Option<Vec<BarrierList>>>,
    event: HANDLE,
    /// Separate fence event waited on exclusively by `victory-flight` outside
    /// `Host::submitted`.
    flight_event: HANDLE,
    /// At-most-one active GPU flight and title fence signals queued behind it.
    flight: Mutex<flight::FlightState>,
    /// Wakes `victory-flight` whenever a new flight is launched.
    flight_wake: Condvar,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
    /// Zero-filled 64 KiB read fallback for null or unmapped direct root CBVs/SRVs.
    zero_root_read: OnceLock<Option<(ID3D12Resource, u64)>>,
    /// Isolated 64 KiB write sink for null or unmapped direct root UAVs.
    sink_root_uav: OnceLock<Option<(ID3D12Resource, u64)>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on, and the mark's mapped
// buffers only used, under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// What shows that a submission ran: a list run last in every submission,
/// which copies the submission's number from upload memory into read-back
/// memory. Under D3DMetal, macOS ends a command buffer when the GPU locks up
/// on it (`kIOGPUCommandBufferCallbackErrorHang`: about 50 ms into a draw
/// whose every pixel runs long) or takes a page fault in it ("MMU
/// interrupt" in the gpuEvent report, VIctory-macOS #178), and after some
/// of those the queue ignores everything submitted to it, presents included
/// (`kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`), while its fence
/// still completes and the device is not removed (VIctory-macOS #50). A
/// list after the ended one in the same submission does not run either.
/// `WriteBufferImmediate` writes nothing under D3DMetal 4.0 beta 3, so this
/// is a copy.
struct Mark {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
    _buffers: [ID3D12Resource; 2],
    /// The number the list copies (upload memory).
    number: *mut u64,
    /// Where it lands (read-back memory).
    seen: *const u64,
}

impl Mark {
    fn new(host: &Host) -> windows::core::Result<Mark> {
        let (number, numbers) = host
            .mapped_buffer(8, false)
            .ok_or_else(windows::core::Error::empty)?;
        let (seen, seen_at) = host
            .mapped_buffer(8, true)
            .ok_or_else(windows::core::Error::empty)?;
        // SAFETY: plain object creation, and a copy between our buffers,
        // which stay in their states (copy source, copy destination).
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            list.CopyBufferRegion(&seen, 0, &number, 0, 8);
            list.Close()?;
            name(&list, || "submission mark".into());
            Ok(Mark {
                list,
                _allocator: allocator,
                _buffers: [number, seen],
                number: numbers.cast(),
                seen: seen_at.cast_const().cast(),
            })
        }
    }

    /// Sets the number the next run of the list copies. Under the
    /// submission lock, once the last submission has finished.
    fn arm(&self, value: u64) {
        // SAFETY: our mapped upload buffer, which no submission reads now.
        unsafe { self.number.write_volatile(value) }
    }

    /// Whether the last run of the list copied `value`.
    fn saw(&self, value: u64) -> bool {
        // SAFETY: our mapped read-back buffer.
        unsafe { self.seen.read_volatile() == value }
    }

    fn list(&self) -> Option<ID3D12CommandList> {
        self.list.cast().ok()
    }
}

/// A closed command list holding a single global UAV barrier, inserted between
/// coalesced queue pieces that crossed a bottom-of-pipe wait or fence boundary
/// so GPU writes in the producer are visible to the consumer.
struct BarrierList {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
}

impl BarrierList {
    fn new(host: &Host) -> windows::core::Result<Self> {
        // SAFETY: plain D3D12 object creation with a single global UAV barrier.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            list.ResourceBarrier(&[uav_barrier()]);
            list.Close()?;
            name(&list, || "coalesced piece UAV barrier".into());
            Ok(Self { list, _allocator: allocator })
        }
    }

    fn list(&self) -> Option<ID3D12CommandList> {
        self.list.cast().ok()
    }
}

/// Submissions that did not run to their mark, and queues made since to
/// replace one that stopped running work.
static LOST: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Lanes that gave up waiting for a 64-bit atomic's lock, over the run
/// (`crate::shadow::take_gave_up`).
static GAVE_UP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Of the preparing's copies (`PREPARING[0..2]`): title copies between
/// buffers (count, microseconds, bytes), then fills of upload memory from
/// title memory (the same), since the last frame-rate line. The rest of
/// those copies are refreshes of host copies.
static COPY_KINDS: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Of the preparing's list steps (`HostList::run_deferred`), since the last
/// frame-rate line, by [`STEP_NAMES`]: how many ran, and the microseconds
/// of the kinds timed (one clock read each: the many refreshes and slots
/// are only counted, their time is the rest of the phase).
static STEPS: [std::sync::atomic::AtomicU64; 2 * STEP_NAMES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2 * STEP_NAMES.len()];

const STEP_NAMES: [&str; 8] = [
    "title copies and fills",
    "refreshes asked",
    "table slots",
    "host work",
    "shader writes declared (snapshots, the refreshes they flush)",
    "table textures",
    "heap scans (bindless)",
    "page tables (title memory)",
];

/// Notes a list step when dropped (each arm of `run_deferred` ends its own way).
struct StepTimer(usize, Option<std::time::Instant>);

impl Drop for StepTimer {
    fn drop(&mut self) {
        note_step(self.0, self.1);
    }
}

/// Counts a list step of kind `kind`, and its microseconds if timed.
fn note_step(kind: usize, started: Option<std::time::Instant>) {
    STEPS[kind].fetch_add(1, Ordering::Relaxed);
    if let Some(started) = started {
        STEPS[STEP_NAMES.len() + kind].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Of the preparing, by phase ([`Phase`]), in microseconds since the last
/// frame-rate line: where a submission's CPU time goes, so a frame report
/// says which part to make faster. Read with [`take_phases`].
static PREP_PHASES: [std::sync::atomic::AtomicU64; PHASE_NAMES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; PHASE_NAMES.len()];

/// The phases of [`Host::prepare_submission`], in the order they run.
#[derive(Clone, Copy)]
enum Phase {
    /// Host textures given up, unordered views made again.
    Evicting,
    /// `descriptor::sync_heaps`: what the title wrote into its heaps.
    HeapSync,
    /// Spare lists trimmed, textures and heaps brought up (`bindless::uploads`).
    Uploads,
    /// The lists' deferred steps (`HostList::run_deferred`): title copies,
    /// the refreshes they flush, heap and page-table scans, host work.
    ListSteps,
    /// The lists' preludes (state moves, hoisted clamps).
    Preludes,
    /// The refreshes still pending once every list's steps ran.
    Refreshes,
    /// Waiting for a flight to land first (`VICTORY_PREP_AHEAD=1`).
    LandingEarly,
    /// Everything else: barrier lists, bookkeeping, the hang note.
    Other,
}

const PHASE_NAMES: [&str; 8] = [
    "evicting",
    "heap views synced",
    "uploads",
    "list steps",
    "preludes",
    "refreshes",
    "landing early",
    "other",
];

/// Adds each phase's time to [`PREP_PHASES`] as the next one starts.
struct PhaseClock(std::time::Instant);

impl PhaseClock {
    /// Counts the time since the last lap as `phase`'s.
    fn lap(&mut self, phase: Phase) {
        let now = std::time::Instant::now();
        PREP_PHASES[phase as usize]
            .fetch_add((now - self.0).as_micros() as u64, Ordering::Relaxed);
        self.0 = now;
    }
}

/// The phases' microseconds since the last call, by [`PHASE_NAMES`].
fn take_phases() -> [u64; PHASE_NAMES.len()] {
    PREP_PHASES.each_ref().map(|c| c.swap(0, Ordering::Relaxed))
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            crate::bench::mark_device_removed();
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => {
                text.push_str("; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command")
            }
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // The frame recorder (`VICTORY_RECORD`, docs/replay.md), before
        // anything is made with the device: it sees what the device makes.
        crate::replay::record::from_env(&device);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool()
            // D3DMetal reports the test and does not apply it (Metal has
            // none): bounds that should reject a draw leave it whole, as when
            // the host has no test, and the plain pipeline states are safer.
            && !crate::under_wine();
        let mut options12 = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        let enhanced_barriers_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut options12).cast(),
                size_of_val(&options12) as u32,
            )
            .is_ok()
            && options12.EnhancedBarriersSupported.as_bool();
        tracing::info!(
            enhanced_barriers = enhanced_barriers_supported,
            "host enhanced barriers"
        );
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let flight_event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        crate::ambient::announce();
        crate::ray_skip::announce();
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match victory_shader::emit::host::Host::from_device(device.clone()).fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            queue: Mutex::new(queue),
            depth_bounds_supported,
            enhanced_barriers_supported,
            fence,
            submitted: Mutex::new(0),
            mark: OnceLock::new(),
            mark_even: OnceLock::new(),
            split_marks: Mutex::new(Some(Vec::new())),
            barrier_lists: Mutex::new(Some(Vec::new())),
            event,
            flight_event,
            flight: Mutex::new(flight::FlightState::default()),
            flight_wake: Condvar::new(),
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
            zero_root_read: OnceLock::new(),
            sink_root_uav: OnceLock::new(),
        })
    }
}

/// Process private commit and working set, in bytes.
pub(crate) fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.PagefileUsage as u64, c.WorkingSetSize as u64)
}

/// Whether the host D3D12 device has been removed.
pub(crate) fn device_removed() -> bool {
    // SAFETY: plain device query.
    host().is_some_and(|h| unsafe { h.device.GetDeviceRemovedReason() }.is_err())
}

/// The PCI vendor id of the host device's adapter (NVIDIA 0x10DE), 0 if it
/// cannot be told.
pub(crate) fn host_vendor() -> u32 {
    static VENDOR: OnceLock<u32> = OnceLock::new();
    *VENDOR.get_or_init(|| {
        let Some(host) = host() else { return 0 };
        // SAFETY: plain DXGI queries for the device's own adapter.
        let vendor = unsafe {
            let luid = host.device.GetAdapterLuid();
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(luid))
                .and_then(|a| a.GetDesc1())
                .map_or(0, |d| d.VendorId)
        };
        tracing::info!(vendor = format_args!("{vendor:#x}"), "host adapter vendor");
        vendor
    })
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: GPU time per kind of command and
/// pipeline (ms, commands) and lists' wall time against the sum of their
/// commands' (ms), since the last frame-rate line.
#[cfg(feature = "diagnostics")]
type GpuTimes = (std::collections::HashMap<String, (f64, u64)>, f64, f64);
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// Adds a timed list's commands (from [`HostList::slowest`]) to [`GPU_TIMES`],
/// by the command's kind and pipeline: its note before the first ` | `,
/// without counts and addresses.
#[cfg(feature = "diagnostics")]
fn gpu_times(wall: Duration, commands: &[(f64, String)]) {
    let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, wall_ms, sum) = times.get_or_insert_with(Default::default);
    *wall_ms += wall.as_secs_f64() * 1e3;
    for (ms, note) in commands {
        // "draw 3x1 [mesh 1f.., pixel 2e..] | ...": the kind and the pipeline.
        let head = note.split(" | ").next().unwrap_or_default();
        let kind = head.split(' ').next().unwrap_or_default();
        let key = match head.find('[') {
            Some(at) => format!("{kind} {}", &head[at..]),
            None => kind.to_owned(),
        };
        let e = by.entry(key).or_default();
        (e.0, e.1) = (e.0 + ms, e.1 + 1);
        *sum += ms;
    }
}

/// Logs [`GPU_TIMES`] per frame over `frames` frames: the 25 costliest.
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let Some((by, wall, sum)) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!(
        "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame",
        wall / frames,
        sum / frames
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (key, (ms, n)) in by.iter().take(25) {
        tracing::info!("  GPU {:.2} ms/frame in {:.1} commands/frame: {key}", ms / frames, *n as f64 / frames);
    }
}

fn video_memory_bytes() -> Option<[(u64, u64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let device = &host()?.device;
            // SAFETY: plain device and DXGI queries.
            unsafe {
                let luid = device.GetAdapterLuid();
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid(luid).ok()
            }
        })
        .as_ref()?;
    let info = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a valid out pointer.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage, info.Budget))
    };
    Some([
        info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?,
        info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
    ])
}

/// Diagnostics: the host device's video memory use and budget, in MB, local
/// (on the card) and non-local (system memory the GPU reads), as the
/// system's memory manager sees them: use over budget makes it page
/// allocations at submissions.
pub(crate) fn video_memory() -> Option<[(f64, f64); 2]> {
    let [(lu, lb), (nu, nb)] = video_memory_bytes()?;
    Some([
        (lu as f64 / 1e6, lb as f64 / 1e6),
        (nu as f64 / 1e6, nb as f64 / 1e6),
    ])
}

/// Local adapter VRAM usage and budget in bytes, for benchmark telemetry.
pub(crate) fn bench_video_memory() -> Option<(u64, u64)> {
    Some(video_memory_bytes()?[0])
}

#[cfg(test)]
pub(crate) mod tests {
    #[test]
    fn mesh_support_requires_both_feature_queries_to_confirm_support() {
        use super::MeshSupport;

        for (mesh_shaders, shader_model_6_6, supported) in [
            (Some(true), Some(true), true),
            (Some(false), Some(true), false),
            (Some(true), Some(false), false),
            (None, Some(true), false),
            (Some(true), None, false),
        ] {
            let features = MeshSupport {
                mesh_shaders,
                shader_model_6_6,
            };
            assert_eq!(features.supported(), supported);
        }
    }

    /// Run in isolation with VICTORY_PRESENT_MAX=1280x720. Exercise the
    /// real swap chain and its resize path without opening a visible window.
    #[test]
    #[ignore = "requires DXGI and an explicit 720p presentation cap"]
    fn capped_presentation_resizes_a_hidden_swap_chain() {
        use super::*;
        let host = host().expect("a host GPU");
        assert_eq!(crate::composite::output_size(1920, 1080), (1280, 720));
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: a hidden Win32 control owned by this test thread.
        let hwnd = unsafe {
            CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("VIctory display test"),
                WS_OVERLAPPEDWINDOW, 0, 0, 320, 180, None, None, None, None)
        }.unwrap();
        let presenter = host.presenter_for_window(&host.queue(), hwnd, 320, 180, format).unwrap();
        host.presenter.lock().unwrap().presenter = Some(presenter);
        let state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
        let picture = crate::composite::result_texture(&host.device, 1920, 1080, format).unwrap();
        let ui = crate::composite::result_texture(&host.device, 3840, 2160, format).unwrap();
        let small = crate::composite::result_texture(&host.device, 640, 360, format).unwrap();
        for (frame, (source, overlay, expected)) in [
            (&picture, None, (1280, 720)),
            (&picture, Some((&ui, state)), (1280, 720)),
            (&small, None, (640, 360)),
        ].into_iter().enumerate() {
            assert!(host.present(source, state, overlay, frame as u64 + 1).is_some());
            let presentation = host.presenter.lock().unwrap();
            let chain = &presentation.presenter.as_ref().unwrap().swap_chain;
            // SAFETY: a live swap chain after presenting and waiting for the copy.
            let desc = unsafe { chain.GetDesc1() }.unwrap();
            assert_eq!((desc.Width, desc.Height), expected);
            // SAFETY: plain query after executing the presentation command list.
            unsafe { host.device.GetDeviceRemovedReason() }.unwrap();
            eprintln!("presented frame {} on a {}x{} DXGI swap chain", frame + 1, desc.Width, desc.Height);
        }
        host.presenter.lock().unwrap().presenter.take();
        // SAFETY: the swap chain was released and this thread owns the window.
        unsafe { DestroyWindow(hwnd) }.unwrap();
    }

    #[derive(Default)]
    struct NativeCallTrace {
        groups: Vec<(Vec<usize>, bool)>,
        calls: Vec<usize>,
    }

    thread_local! {
        static NATIVE_CALLS: std::cell::RefCell<Option<NativeCallTrace>> =
            const { std::cell::RefCell::new(None) };

        /// Every transition barrier built on this thread (resource, before,
        /// after), in order.
        pub(crate) static ISSUED: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The barriers built on this thread since the last call.
    pub(crate) fn issued() -> Vec<(usize, u32, u32)> {
        ISSUED.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    pub(crate) fn start_native_call_trace() {
        NATIVE_CALLS.with(|trace| {
            assert!(trace.borrow_mut().replace(NativeCallTrace::default()).is_none());
        });
    }

    pub(crate) fn take_native_call_trace() -> Vec<(Vec<usize>, bool)> {
        NATIVE_CALLS.with(|trace| trace.borrow_mut().take().unwrap().groups)
    }

    pub(super) fn native_call(lists: usize) {
        NATIVE_CALLS.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.calls.push(lists);
            }
        });
    }

    pub(super) fn native_call_group(fence_waited: bool) {
        NATIVE_CALLS.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.groups.push((std::mem::take(&mut trace.calls), fence_waited));
            }
        });
    }

    /// Replays transitions in the order the GPU runs them from the states
    /// in `states` (resources not in it are not checked), by the debug
    /// layer's rule: each barrier's before-state is the resource's current
    /// one. The first that is not, else the states after.
    pub(crate) fn replay(issued: &[(usize, u32, u32)], states: &mut std::collections::HashMap<usize, u32>) -> Result<(), String> {
        for (k, &(resource, before, after)) in issued.iter().enumerate() {
            if let Some(state) = states.get_mut(&resource) {
                if *state != before {
                    return Err(format!("barrier {k} on {resource:#x}: before {before:#x}, but the resource is in {state:#x}"));
                }
                *state = after;
            }
        }
        Ok(())
    }

    #[test]
    fn frame_report_includes_the_frames_before_the_last_caption() {
        use super::FrameWindow;
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let sample = |seconds, frames, drawn| FrameWindow {
            since: start + Duration::from_secs(seconds),
            frames,
            drawn,
        };
        let mut caption = sample(0, 0, 10);
        let mut report = caption;
        let second = Duration::from_secs(1);
        let period = Duration::from_secs(5);
        let early = sample(4, 120, 370);
        assert_eq!(caption.sample(early, second).unwrap().fps, 30.0);
        assert!(report.sample(early, period).is_none());

        let end = sample(5, 125, 510);
        let short = caption.sample(end, second).unwrap();
        let full = report.sample(end, period).unwrap();
        assert_eq!((short.fps, short.ms, short.draws), (5.0, 200.0, 28.0));
        assert_eq!((full.frames, full.fps, full.ms, full.draws), (125.0, 25.0, 40.0, 4.0));
        assert!(report.sample(end, period).is_none());
        let following = report.sample(sample(10, 135, 530), period).unwrap();
        assert_eq!((following.frames, following.fps, following.draws), (10.0, 2.0, 2.0));
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    /// Shaders that use title memory mark their reads in their own bitmap:
    /// read marks of both bitmaps are taken, each slot once until it is
    /// forgotten, and a forgotten slot's next read is taken again.
    #[test]
    fn read_marks_of_both_bitmaps_are_taken() {
        let Some(host) = host() else { return };
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let (plain, memory) = (victory_shader::emit::HEAP_VIEWS - 2, victory_shader::emit::HEAP_VIEWS - 1);
        crate::shadow::tests::mark_read(false, plain);
        crate::shadow::tests::mark_read(true, memory);
        let taken = crate::shadow::take_reads();
        assert!(taken.contains(&plain), "a plain shader's read");
        assert!(taken.contains(&memory), "a read by a shader that uses title memory");
        let again = crate::shadow::take_reads();
        assert!(!again.contains(&plain) && !again.contains(&memory), "each once");
        crate::shadow::forget_read(memory);
        assert!(!crate::shadow::take_reads().contains(&memory), "forgotten: its mark cleared");
        crate::shadow::tests::mark_read(true, memory);
        assert!(crate::shadow::take_reads().contains(&memory), "read again after it was forgotten");
    }

    /// The textures shaders read through the heap since the last eviction
    /// pass are named again at each pass (a slot's read is reported as new
    /// only once), and the marks are cleared for the next.
    #[test]
    fn heap_reads_since_the_last_pass_name_their_textures() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(texture).unwrap().texture) as usize;
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..2 {
            crate::shadow::tests::mark_read(false, 0);
            let slots = crate::shadow::reads_since();
            assert!(slots.contains(&0), "the slot read");
            assert!(crate::descriptor::read_textures(&slots).contains(&raw), "its texture named");
            assert!(!crate::shadow::reads_since().contains(&0), "the marks cleared");
        }
    }

    /// While [`ShowAfterScene`] lives, this thread takes `VICTORY_SHOW_AFTER`
    /// as `on` and holds the submission lock, so the submission counters
    /// change only as the test says (one submission begun, then run); both
    /// are put back when it goes.
    struct ShowAfterScene<'a> {
        // Dropped before the lock: what is made under it (a view heap) does
        // not take the lock again (`Host::quiet`) while this is held.
        _holding: Holding,
        _one: std::sync::MutexGuard<'a, u64>,
        done: u64,
    }

    impl ShowAfterScene<'_> {
        fn new(host: &Host, on: bool) -> ShowAfterScene<'_> {
            let one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let holding = Holding::enter();
            crate::descriptor::host::set_show_after(Some(on));
            let done = SUBMISSIONS_DONE.load(Ordering::Acquire);
            SUBMISSIONS_BEGUN.store(done, Ordering::Release);
            ShowAfterScene { _holding: holding, _one: one, done }
        }

        /// A submission is handed to the queue and still runs.
        fn begin(&self) {
            SUBMISSIONS_BEGUN.store(self.done + 1, Ordering::Release);
        }

        /// That submission has run: none runs any more.
        fn finish(&self) {
            SUBMISSIONS_DONE.store(self.done + 1, Ordering::Release);
        }
    }

    impl Drop for ShowAfterScene<'_> {
        fn drop(&mut self) {
            // Back to `*submitted`, which the lock kept as it was.
            SUBMISSIONS_DONE.store(self.done, Ordering::Release);
            SUBMISSIONS_BEGUN.store(self.done, Ordering::Release);
            crate::descriptor::host::set_show_after(None);
        }
    }

    /// A title heap of 4 view slots with a 12x1 RGBA8 placed texture viewed
    /// at slot 0, made in it; and a way to write a view of another texture
    /// of the same memory there (`n` 1, 2, ...: its offset), as the title
    /// would. Placed textures are kept by address and description for the
    /// process (`resource::texture_at`), so the memory is never freed and
    /// the width is one no other test uses: a texture another test made at
    /// a reused address would not be new.
    fn show_after_heap() -> (&'static crate::descriptor::Heap, impl Fn(u64)) {
        let memory: &'static [u8] = Vec::leak(vec![0u8; 1 << 17]);
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let base = memory.as_ptr() as u64;
        let start = title_heap.start;
        let write = move |n: u64| {
            let words: [u32; 14] = [3, 0, 0, 0, 12, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
            let address = base + n * (1 << 13);
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
        };
        write(0);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, start, true).is_some());
        assert!(heap_host.texture_slot_made_from(0, start), "the first texture shown");
        (title_heap, write)
    }

    /// `VICTORY_SHOW_AFTER=1`: a slot whose new record views a texture made
    /// while a submission runs keeps showing what it showed (lists on the
    /// GPU read it by index), and is made from the new record at the next
    /// submission's preparation, when none runs.
    #[test]
    fn show_after_holds_a_slot_of_a_fresh_texture_until_the_next_submission() {
        let Some(host) = host() else { return };
        let scene = ShowAfterScene::new(host, true);
        let (title_heap, write) = show_after_heap();
        let heap_host = title_heap.host().expect("a host view heap");
        scene.begin();
        write(1);
        assert!(heap_host.mirror(0, title_heap.start, true).is_some(), "the new view, for the draw's list");
        assert!(heap_host.is_held(0), "held while the submission runs");
        assert!(!heap_host.texture_slot_made_from(0, title_heap.start), "the slot still shows the first texture");
        scene.finish();
        assert_eq!(heap_host.show_held(), 1);
        assert!(!heap_host.is_held(0));
        assert!(heap_host.texture_slot_made_from(0, title_heap.start), "the new texture shown");
    }

    /// Without the switch a slot is made at once, as before.
    #[test]
    fn show_after_off_makes_a_slot_of_a_fresh_texture_at_once() {
        let Some(host) = host() else { return };
        let scene = ShowAfterScene::new(host, false);
        let (title_heap, write) = show_after_heap();
        let heap_host = title_heap.host().expect("a host view heap");
        scene.begin();
        write(1);
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        assert!(!heap_host.is_held(0));
        assert!(heap_host.texture_slot_made_from(0, title_heap.start), "made at once");
    }

    /// A held slot the title wrote again before the next submission is not
    /// made from the record it was held for: that write is the slot's.
    #[test]
    fn show_after_drops_a_held_slot_the_title_wrote_again() {
        let Some(host) = host() else { return };
        let scene = ShowAfterScene::new(host, true);
        let (title_heap, write) = show_after_heap();
        let heap_host = title_heap.host().expect("a host view heap");
        scene.begin();
        write(1);
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        assert!(heap_host.is_held(0));
        write(2);
        scene.finish();
        assert_eq!(heap_host.show_held(), 0, "the record held for is gone");
        assert!(!heap_host.is_held(0));
    }

    /// The idle rule against the heap reads that thrashed (run evict1): a
    /// texture shaders read through the heap in every frame stays over more
    /// frames and time than the idle limits; one read once is given up. A
    /// list of the title's naming a texture (a table read, a copy) stamps it
    /// as used; a list of ours (uploads) does not.
    #[test]
    fn textures_read_every_frame_stay_and_title_lists_stamp_theirs() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let step = 32;
        // Two 4x1 RGBA8 textures, linear, in memory of their own, viewed at
        // slots 5 and 6.
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let heap_host = title_heap.host().expect("a host view heap");
        let textures: Vec<crate::object::This> = [5u32, 6]
            .iter()
            .map(|&slot| {
                let address = memory.as_ptr() as u64 + u64::from(slot) * (1 << 13);
                let mut view = [0u32; 16];
                view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                let at = title_heap.start + u64::from(slot) * step;
                crate::descriptor::create_placed_srv(null, &words, &view, at);
                assert!(heap_host.mirror(slot, at, true).is_some());
                crate::resource::texture_at(address, &words).expect("its resource")
            })
            .collect();
        let owned: Vec<_> = textures.iter().map(|&t| crate::resource::owned(t).expect("may be given up")).collect();
        let (idle_frames, idle_ms) = (crate::resource::IDLE_FRAMES, crate::resource::IDLE_MS);
        {
            // No submission takes the marks meanwhile.
            let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Passes a third of the idle limits apart, over three times them.
            for k in 0..10 {
                crate::shadow::tests::mark_read(false, 5);
                if k == 0 {
                    crate::shadow::tests::mark_read(false, 6);
                }
                let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
                crate::resource::evict_among(&owned, &read, 1000 + k * idle_frames / 3, 1_000_000 + k * idle_ms / 3, 0);
                assert!(crate::resource::made_copy(textures[0]).is_some(), "read every frame: stays (frame {k})");
            }
        }
        assert!(crate::resource::made_copy(textures[1]).is_none(), "read once: given up");
        drop(owned);
        // A title list naming the texture stamps it, an upload list does not.
        let copy = crate::resource::host_copy(textures[0]).unwrap();
        for (ours, stamped) in [(true, false), (false, true)] {
            copy.unstamp();
            let list = host.new_list().unwrap();
            if ours {
                list.set_ours();
            }
            let _ = list.transition(textures[0], 0x40);
            assert_eq!(copy.last_used().0 != u64::MAX, stamped, "ours: {ours}");
        }
    }

    /// Textures the heaps view are brought up to date only if the title used
    /// them lately: one whose pages changed but that no shader read is not
    /// uploaded; read in the last pass, it is.
    #[test]
    fn heap_textures_unread_lately_are_not_brought_up() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(null, address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        viewed.set(0, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        copy.unstamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 0, "not read lately: not brought up");
        copy.stamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 1, "read in the last pass: brought up");
    }

    /// Heap textures brought up in one submission are refreshed side by
    /// side, each into its own staging and its own upload.
    #[test]
    fn heap_textures_brought_up_together_each_get_their_upload() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let pool = crate::watch::tests::pool(6);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let objects: Vec<crate::object::This> = (0..6u32)
            .map(|k| {
                // RGBA8 128x128 (one page each), linear, each its own bytes.
                let address = (pool + k as usize * crate::watch::PAGE) as u64;
                // SAFETY: the texture's own page of the pool.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x10 * k as u8, crate::watch::PAGE) };
                let words: [u32; 14] = [3, 0, 0, 0, 128, 0, 128, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                let mut object = std::ptr::null();
                assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
                crate::resource::host_copy(object).expect("a host texture").stamp();
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                viewed.set(k, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
                object
            })
            .collect();
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "each brought up once");
        }
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "unchanged: not again");
        }
    }

    #[test]
    fn list_scratch_grows_to_fit_and_keeps_what_the_last_cycle_reached() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let chunks = || list.scratch.lock().unwrap().chunks.len();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        let wide = list.scratch(200 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[2].capacity, 256 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        // The cycle reached the third chunk: all kept, at their addresses.
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch(200 << 10).unwrap(), wide);
        assert_eq!(chunks(), 3);
        list.close();
        host.execute(&[&list]);
        list.reset();
        // One that reached the second: the third stays (one chunk of slack).
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(chunks(), 3);
        // One that stayed in the first: the third goes (the old rule kept
        // every chunk for good), the second stays.
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(chunks(), 2);
        assert_eq!(list.scratch(512).unwrap(), first);
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        list.close();
    }

    #[test]
    fn list_uploads_share_scratch_chunks_at_texture_alignment() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let (first, at, _) = list.upload(1).unwrap();
        assert_eq!(at, 0);
        // A 256-byte scratch use in between leaves the next upload at 512.
        let (gpu, _) = list.scratch(1).unwrap();
        let (second, next, _) = list.upload(700).unwrap();
        assert_eq!(next, 512);
        assert_eq!(first, second, "one chunk, not a buffer per copy");
        // SAFETY: a live buffer.
        assert_eq!(gpu, unsafe { first.GetGPUVirtualAddress() } + 256);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        // A copy larger than a chunk has a buffer of its own, at 0.
        let (big, at, _) = list.upload(SCRATCH_CHUNK + 1).unwrap();
        assert_eq!(at, 0);
        assert_ne!(big.as_raw(), first.as_raw());
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        list.close();
    }

    #[test]
    fn a_shed_list_makes_its_scratch_again_from_the_first_chunk() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        list.scratch(200 << 10).unwrap();
        list.close();
        list.reset();
        assert!(list.scratch_bytes() > 0, "kept for the next cycle");
        list.shed_scratch();
        assert_eq!(list.scratch_bytes(), 0);
        list.scratch(1).unwrap();
        assert_eq!(list.scratch_bytes(), FIRST_SCRATCH_CHUNK);
        list.close();
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }

    #[test]
    fn parallel_batch_flush_updates_multiple_shadows_identically() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if super::host().is_none() {
            return;
        }
        let len = 0x40000usize;
        let mut ptrs = Vec::new();
        let mut shadows = Vec::new();
        for i in 0..3usize {
            // SAFETY: fresh committed allocation, freed below.
            let p = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
            assert_ne!(p, 0);
            let pattern = (0x21 + i as u8).wrapping_mul(3);
            // SAFETY: `len` committed bytes at `p`.
            unsafe { std::ptr::write_bytes(p as *mut u8, pattern, len) };
            let id = 0x9100 + i;
            crate::shadow::register(id, p as u64, len as u64);
            let shadow = crate::shadow::find(p as u64, len as u64).expect("shadow");
            ptrs.push((id, p, pattern));
            shadows.push(shadow);
        }
        let mut batch = super::Batch::default();
        for (i, &(_, p, _)) in ptrs.iter().enumerate() {
            batch.add(&shadows[i], p, len);
        }
        let copied = batch.flush();
        assert!(copied >= 3 * len);
        for (i, &(_, _, pattern)) in ptrs.iter().enumerate() {
            assert_eq!(shadows[i].read_host(ptrs[i].1 as u64, 4), vec![u32::from_le_bytes([pattern; 4]); 4]);
            crate::shadow::unregister(ptrs[i].0, ptrs[i].1 as u64);
            // SAFETY: our allocation.
            unsafe { VirtualFree(ptrs[i].1 as *mut _, 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn bind_pipeline_deduplicates_kept_objects_until_reset() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let Ok((root, pso)) = crate::compute::pipeline(
            "[numthreads(1, 1, 1)] void main() {}",
            windows::core::s!("bind_pipeline_test"),
            0,
            &[],
        ) else {
            list.close();
            return;
        };

        list.bind_pipeline("test_pso", &pso, &root);
        list.bind_pipeline("test_pso", &pso, &root);
        assert_eq!(list.deferred().kept.len(), 2);
        assert_eq!(list.deferred().bound_pipelines.len(), 1);

        list.close();
        list.reset();
        assert!(list.deferred().kept.is_empty());
        assert!(list.deferred().bound_pipelines.is_empty());
        list.bind_pipeline("test_pso", &pso, &root);
        assert_eq!(list.deferred().kept.len(), 2);
        list.close();
    }

    #[test]
    fn keep_deduplicates_repeated_objects_until_reset() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let Some((a, _)) = host.mapped_buffer(256, false) else {
            list.close();
            return;
        };
        let Some((b, _)) = host.mapped_buffer(256, false) else {
            list.close();
            return;
        };

        list.keep(&a);
        list.keep(&a);
        assert_eq!(list.deferred().kept.len(), 1);
        assert_eq!(list.deferred().kept_ptrs.len(), 1);

        list.keep(&b);
        assert_eq!(list.deferred().kept.len(), 2);
        assert_eq!(list.deferred().kept_ptrs.len(), 2);

        list.close();
        list.reset();
        assert!(list.deferred().kept.is_empty());
        assert!(list.deferred().kept_ptrs.is_empty());

        list.keep(&a);
        assert_eq!(list.deferred().kept.len(), 1);
        assert_eq!(list.deferred().kept_ptrs.len(), 1);
        list.close();
    }

    #[test]
    fn fill_scratch_reuses_upload_until_title_copy_and_reset() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let data = [0x11u8; 64];
        let a = list.fill_scratch(data.as_ptr(), 64).expect("first fill");
        let b = list.fill_scratch(data.as_ptr(), 64).expect("cached fill");
        assert_eq!(a, b);
        assert_eq!(list.deferred().copies.len(), 1);

        let mut dst = [0u8; 16];
        list.defer_copy(dst.as_mut_ptr(), data.as_ptr(), 16);
        let c = list.fill_scratch(data.as_ptr(), 64).expect("after title copy");
        assert_ne!(a, c);
        assert_eq!(list.deferred().copies.len(), 3);

        list.close();
        list.reset();
        let d = list.fill_scratch(data.as_ptr(), 64).expect("after reset");
        assert_eq!(d, a, "scratch restarts from chunk 0 after reset");
        list.close();
    }

    #[test]
    fn host_list_epoch_and_deferred_steps_track_mutations() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let start_epoch = list.epoch();
        let start_def_epoch = list.deferred_epoch();
        assert_ne!(start_epoch, 0);
        assert_ne!(start_def_epoch, 0);
        assert_eq!(list.deferred_steps(), 0);

        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, crate::watch::PAGE as u64);
        let shadow = crate::shadow::find(p as u64, crate::watch::PAGE as u64).expect("shadow");

        // First refresh queues 1 deferred step without invalidating other
        // already-recorded read entries; identical repeat adds no step.
        list.defer_refresh(&shadow, p, 256);
        assert_eq!(list.epoch(), start_epoch);
        assert_eq!(list.deferred_steps(), 1);

        list.defer_refresh(&shadow, p, 256);
        assert_eq!(list.epoch(), start_epoch, "deduplicated refresh keeps epoch");
        assert_eq!(list.deferred_steps(), 1, "deduplicated refresh adds no step");

        // Deferred copy bumps both epochs, adds a step, and clears refresh dedup.
        let src = [0x22u8; 16];
        list.defer_copy(p as *mut u8, src.as_ptr(), src.len());
        let after_copy = list.epoch();
        let after_copy_def = list.deferred_epoch();
        assert_ne!(after_copy, start_epoch);
        assert_ne!(after_copy_def, start_def_epoch);
        assert_eq!(list.deferred_steps(), 2);

        // need_rebind and defer_gpu_write do not invalidate already-recorded
        // read-only descriptor preparation.
        list.need_rebind();
        assert_eq!(list.epoch(), after_copy);
        assert_eq!(list.deferred_epoch(), after_copy_def);
        assert!(list.take_rebind());

        list.defer_gpu_write(&shadow, p, 256, false);
        assert_eq!(list.epoch(), after_copy);
        assert_eq!(list.deferred_epoch(), after_copy_def);
        assert_eq!(list.deferred_steps(), 3);

        // Reset advances both epochs and clears queued deferred steps.
        list.close();
        list.reset();
        assert_ne!(list.epoch(), after_copy);
        assert_ne!(list.deferred_epoch(), after_copy_def);
        assert_eq!(list.deferred_steps(), 0);
        list.close();

        crate::shadow::unregister(p, p as u64);
    }

    #[test]
    fn host_list_will_write_and_batch_flags_reflect_deferred_steps() {
        let Some(host) = host() else { return };
        let Ok(producer) = host.new_list() else { return };
        let Ok(consumer) = host.new_list() else { return };

        let mut word = 0u64;
        let addr = (&raw mut word) as u64;
        let wait_ge_5 = crate::hold::ValueWait {
            address: addr,
            bytes: 8,
            comparison: 7, // GREATER_EQUAL
            reference: 5,
            mask: u64::MAX,
        };
        let wait_ge_9 = crate::hold::ValueWait {
            address: addr,
            bytes: 8,
            comparison: 7,
            reference: 9,
            mask: u64::MAX,
        };

        assert!(!producer.will_write(&wait_ge_5));
        producer.defer_write(addr as *mut u8, crate::queue::Written::Value(7), 8);
        assert!(producer.will_write(&wait_ge_5));
        assert!(!producer.will_write(&wait_ge_9));
        // A later timestamp write to the same address is unknown at submission time.
        producer.defer_write(addr as *mut u8, crate::queue::Written::GpuTimestamp, 8);
        assert!(!producer.will_write(&wait_ge_5));
        // A later value write restores a known satisfying value.
        producer.defer_write(addr as *mut u8, crate::queue::Written::Value(9), 8);
        assert!(producer.will_write(&wait_ge_9));

        assert_eq!(producer.batch_flags(), crate::hold::BatchFlags::default());
        let src = [0x22u8; 16];
        let mut dst = [0u8; 16];
        producer.defer_copy(dst.as_mut_ptr(), src.as_ptr(), 16);
        let p_flags = producer.batch_flags();
        assert!(p_flags.has_title_copy && p_flags.has_cpu_read);
        assert!(!p_flags.has_gpu_write && !p_flags.has_after && !p_flags.has_textures);

        consumer.defer_after(|| {});
        let c_flags = consumer.batch_flags();
        assert!(c_flags.has_after);
        assert!(crate::hold::can_join_batch(p_flags, c_flags, 1, 64));
        assert!(!crate::hold::can_join_batch(c_flags, p_flags, 1, 64));

        producer.close();
        consumer.close();
        host.execute_with_barriers(&[&producer, &consumer], &[false, true]);
        assert_eq!(word, 9);
    }
}

/// Monotonic command-list mutation epoch shared across all [`HostList`]s so
/// resets and newly created lists never reuse an earlier epoch.
static NEXT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_epoch() -> u64 {
    NEXT_EPOCH.fetch_add(1, Ordering::Relaxed)
}

type WidenedWrite = (std::sync::Arc<crate::shadow::Shadow>, usize, usize);

/// UAV writes and heap-writing draws in command-recording order.
enum WidenedEvent {
    Write(WidenedWrite),
    Heap(std::sync::Arc<crate::bindless::Viewed>),
}

/// The position of each heap's last event, by heap address (the events hold
/// the heaps, so no address is reused).
fn last_heap_events(events: &[WidenedEvent]) -> std::collections::HashMap<usize, usize> {
    events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            WidenedEvent::Heap(heap) => Some((std::sync::Arc::as_ptr(heap) as usize, index)),
            WidenedEvent::Write(_) => None,
        })
        .collect()
}

/// A list's widened events since its last barrier, and the writes taken
/// from them that no barrier widened yet (heap events expanded). Both are
/// emptied, not made again, from barrier to barrier and from one use of the
/// list to the next: a heap event expands to every UAV range of the heap,
/// tens of thousands in the world.
#[derive(Default)]
struct WidenedPending {
    events: Vec<WidenedEvent>,
    writes: Vec<WidenedWrite>,
}

/// Room for widened writes a list keeps when it closes at least, in writes
/// (24 bytes each), so that small lists never give theirs back.
const WIDENED_KEEP_MIN: usize = 1 << 12;

/// Empties a list's widened write buffer at its close, keeping the room for
/// its next use (the world's lists need about the same each frame) but not
/// more than twice what this use needed, `used`: a peak is not kept for
/// good.
fn empty_widened<T>(v: &mut Vec<T>, used: usize) {
    v.clear();
    let keep = used.max(WIDENED_KEEP_MIN).next_power_of_two();
    if v.capacity() > 2 * keep {
        v.shrink_to(keep);
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Cached `ID3D12GraphicsCommandList7` for `list` when the host supports
    /// Enhanced Barriers (`D3D12_FEATURE_D3D12_OPTIONS12`), avoiding per-draw
    /// `QueryInterface`.
    pub(crate) list7: Option<ID3D12GraphicsCommandList7>,
    /// Cached `ID3D12GraphicsCommandList7` for `prelude.1` when Enhanced
    /// Barriers are supported.
    pub(crate) prelude7: Option<ID3D12GraphicsCommandList7>,
    /// Monotonic epoch bumped whenever texture state, twin state, deferred
    /// dedup sets, or the list itself resets ([`HostList::changed_something`]).
    epoch: std::sync::atomic::AtomicU64,
    /// Monotonic epoch bumped only when deferred refresh/slot deduplication
    /// sets are cleared (`reset` or `defer_title_copy`).
    deferred_epoch: std::sync::atomic::AtomicU64,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset (`trim_scratch` gives up what a peak grew).
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// Shadow ranges written through UAVs: the history late views may need,
    /// and ordered events written since the last barrier.
    widened_writes: Mutex<Vec<WidenedWrite>>,
    widened_pending: Mutex<WidenedPending>,
    /// Exact source shadows whose buffers widening commands read. Their COM
    /// objects alone do not keep an arena slice out of the reuse pool.
    widened_sources: Mutex<Vec<std::sync::Arc<crate::shadow::Shadow>>>,
    /// Widened buffers that have consumed every write recorded so far.
    widened_ready: Mutex<std::collections::HashSet<usize>>,
    /// Widened buffers this list has transitioned, by host address.
    widened_states: Mutex<
        std::collections::HashMap<usize, (ID3D12Resource, D3D12_RESOURCE_STATES)>,
    >,
    /// A list of ours (uploads), whose uses are no use of the title's
    /// (`crate::resource::evict`).
    ours: std::sync::atomic::AtomicBool,
    /// The draw or dispatch being recorded runs a program using title
    /// memory ([`HostList::writer_uses_memory`]).
    memory_writer: std::sync::atomic::AtomicBool,
    /// A marker group for the bound pipeline is open ([`markers`]).
    marked: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions: shared
    /// with the hang note ([`WAITING_ON`]), which names them only if the
    /// GPU hangs, so a submission takes the sets, not copies of the names.
    pipelines: Mutex<Pipelines>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Clamped indirect arguments (`crate::indirect`), reused by each
    /// inline execution in turn in legacy sync mode; replaced by a bigger one
    /// when too small.
    arguments: Mutex<Option<ID3D12Resource>>,
    /// Bump-allocated disjoint 16-byte-aligned inline argument slices used in
    /// narrow clamp sync mode (`VICTORY_CLAMP_SYNC=1`) so consecutive inline
    /// clamps have zero WAR hazard against earlier `ExecuteIndirect` calls.
    inline_arguments: Mutex<ArgumentArena>,
    /// Indirect clamps hoisted into `prelude`, their bump-allocated argument
    /// buffers, and the list's GPU-write hazard tracker.
    clamps: Mutex<HoistedClamps>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// With `VICTORY_GPU_TIMING_CUTS`: the one kind and pipeline of the
    /// list's draws or dispatches, once it has any (`crate::timing_cuts`).
    #[cfg(feature = "diagnostics")]
    timed: Mutex<Option<crate::timing_cuts::Timed>>,
}

// SAFETY: D3D12 objects are free-threaded; a host list is recorded by the
// title list's one recording thread, then run by one submission at a time
// (the queues hold it between, `crate::hold`).
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States =
    std::collections::HashMap<usize, (u32, u32, std::sync::Arc<crate::resource::HostTexture>)>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING") || timing_cuts_on())
}

/// Whether a few pipelines at a time get host lists of their own, so that
/// a timed list of theirs runs one pipeline and its wall time is that
/// pipeline's (`VICTORY_GPU_TIMING_CUTS`, diagnostics builds; turns the
/// timing on): GPU time per pipeline where timestamps do not advance, as
/// under D3DMetal. See `crate::timing_cuts`.
#[cfg(feature = "diagnostics")]
pub(crate) fn timing_cuts_on() -> bool {
    *timing_cuts() != crate::timing_cuts::Mode::Off
}

#[cfg(feature = "diagnostics")]
fn timing_cuts() -> &'static crate::timing_cuts::Mode {
    static MODE: OnceLock<crate::timing_cuts::Mode> = OnceLock::new();
    MODE.get_or_init(|| {
        let mode = crate::timing_cuts::mode_of(std::env::var("VICTORY_GPU_TIMING_CUTS").ok().as_deref());
        if mode != crate::timing_cuts::Mode::Off {
            tracing::info!(
                ?mode,
                "VICTORY_GPU_TIMING_CUTS: host lists of their own for a few pipelines at a time, each list \
                 run and timed on its own"
            );
        }
        mode
    })
}

/// A sweep's pipelines and where it is (`VICTORY_GPU_TIMING_CUTS=sweep`).
#[cfg(feature = "diagnostics")]
static SWEEP: Mutex<Option<crate::timing_cuts::Sweep>> = Mutex::new(None);

/// Whether the title pipeline `this` gets host lists of its own now: it
/// has a listed shader (remembered per pipeline, which the title makes
/// once and keeps), or the sweep is at it.
#[cfg(feature = "diagnostics")]
pub(crate) fn timing_cut_own(this: usize) -> bool {
    use crate::timing_cuts::Mode;
    if this == 0 {
        return false;
    }
    match timing_cuts() {
        Mode::Off => false,
        Mode::Sweep(_) => SWEEP
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .times(this),
        Mode::Listed(listed) => {
            static SEEN: Mutex<Option<std::collections::HashMap<usize, bool>>> = Mutex::new(None);
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            *seen
                .get_or_insert_with(Default::default)
                .entry(this)
                .or_insert_with(|| {
                    crate::timing_cuts::lists(
                        listed,
                        &crate::pipeline::shaders_of(this as crate::object::This),
                    )
                })
        }
    }
}

/// The timed lists' wall time by pipeline since the last frame-rate line
/// (`VICTORY_GPU_TIMING_CUTS`).
#[cfg(feature = "diagnostics")]
static GPU_WALLS: Mutex<Option<crate::timing_cuts::Walls>> = Mutex::new(None);

/// Logs [`GPU_WALLS`] over `frames` frames: the 40 pipelines that cost most
/// above the lists' floor, each per frame it had lists in.
#[cfg(feature = "diagnostics")]
fn log_gpu_walls(frames: f64) {
    let Some(walls) = GPU_WALLS.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let (lists, ms, floor, rows) = walls.report(frames);
    let sweep = match timing_cuts() {
        crate::timing_cuts::Mode::Sweep(n) => {
            let mut sweep = SWEEP.lock().unwrap_or_else(|e| e.into_inner());
            let (now, live) = sweep.get_or_insert_with(Default::default).advance(*n);
            format!("; the sweep goes on with {now} of the {live} pipelines drawn with")
        }
        _ => String::new(),
    };
    tracing::info!(
        "GPU wall time by pipeline: {lists:.1} timed lists/frame, {ms:.1} ms/frame; a list's floor {floor:.3} ms \
         (tenth percentile); {} lines{sweep}",
        rows.len()
    );
    for r in rows.iter().take(40) {
        tracing::info!(
            "  GPU wall {:.2} ms ({:.2} above the floor) in {:.1} lists, {:.1} commands a frame, \
             in {} frames: {}",
            r.ms,
            r.above,
            r.lists,
            r.commands,
            r.frames,
            r.label
        );
    }
}

/// On D3DMetal, a separate native queue call before a deferred UAV-to-SRV
/// prelude can make the preceding producer's writes visible.
fn split_uav_srv_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE"))
}

/// Whether host lists group each title pipeline's commands in a marker
/// group named by its shaders (`VICTORY_GPU_MARKERS=1`), so that a GPU
/// capture or trace of the host can tell them apart: D3D12 timestamps all
/// read 0 under D3DMetal, so `VICTORY_GPU_TIMING` cannot time them on a
/// Mac. Whether a tool shows the groups is up to the host's driver (on a
/// Mac, D3DMetal's debug groups); off, nothing is recorded.
pub(crate) fn markers() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_MARKERS"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on), and for
/// the frame recorder (`VICTORY_RECORD`), whose reports name pipelines and
/// resources by it.
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        return name_it(object, || text);
    }
    name_it(object, text);
}

/// [`name`] for the device-removed report alone.
fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;
const ARGUMENT_CHUNK_FIRST: u64 = 64 << 10;
const ARGUMENT_CHUNK_MAX: u64 = 4 << 20;
const MAX_HAZARD_RANGES: usize = 256;
/// Clamps a list started for a clamp must hoist before another is cut for
/// ([`HostList::clamp_cut_pays`]): with one, the cut cost what clamping
/// inline would have.
const CUT_PAYS: usize = 2;

struct ArgumentChunk {
    buffer: ID3D12Resource,
    gpu: u64,
    capacity: u64,
}

/// Tracks title-address ranges written on the GPU earlier in a host list so
/// indirect argument clamps only hoist into the list's prelude when neither
/// their argument range nor their count word could have been written earlier
/// in the same list, and preserves sequential budget order once any inline
/// clamp has run.
/// Why an indirect argument clamp cannot run in its list's prelude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClampBlock {
    /// An earlier command of the list may write any title buffer (heap
    /// views, title memory).
    Unknown,
    /// An earlier command of the list writes the arguments or the count.
    Overlap,
    /// An earlier clamp of the list ran inline (the budget is charged in
    /// recorded order).
    AfterInline,
}

#[derive(Default)]
pub(crate) struct WriteHazard {
    /// Sorted, disjoint, non-empty half-open title address ranges `[start, end)`.
    ranges: Vec<(u64, u64)>,
    /// Set when an earlier command can write arbitrary buffer memory (e.g.
    /// bindless heap UAV writes or page-table memory writes).
    unknown: bool,
    /// Set once an inline clamp has been recorded in this list so subsequent
    /// clamps stay inline and charge the shared budget in recorded order.
    inline_used: bool,
}

impl WriteHazard {
    pub(crate) fn note_write(&mut self, start: u64, len: u64) {
        if self.unknown || len == 0 {
            return;
        }
        let end = start.saturating_add(len);
        if end <= start {
            return;
        }
        if self.ranges.last() == Some(&(start, end)) {
            return;
        }
        let pos = self.ranges.partition_point(|&(s, _)| s < start);
        let mut lo = pos;
        let mut new_start = start;
        let mut new_end = end;
        if lo > 0 && self.ranges[lo - 1].1 >= new_start {
            lo -= 1;
            new_start = self.ranges[lo].0;
            new_end = new_end.max(self.ranges[lo].1);
        }
        let mut hi = pos;
        while hi < self.ranges.len() && self.ranges[hi].0 <= new_end {
            new_end = new_end.max(self.ranges[hi].1);
            hi += 1;
        }
        self.ranges.splice(lo..hi, std::iter::once((new_start, new_end)));
        if self.ranges.len() > MAX_HAZARD_RANGES {
            self.unknown = true;
            self.ranges.clear();
        }
    }

    pub(crate) fn note_unknown(&mut self) {
        self.unknown = true;
        self.ranges.clear();
    }

    pub(crate) fn note_inline(&mut self) {
        self.inline_used = true;
    }

    pub(crate) fn overlaps(&self, start: u64, len: u64) -> bool {
        if self.unknown {
            return true;
        }
        if len == 0 {
            return false;
        }
        let end = start.saturating_add(len);
        let pos = self.ranges.partition_point(|&(_, e)| e <= start);
        pos < self.ranges.len() && self.ranges[pos].0 < end
    }

    #[cfg(test)]
    pub(crate) fn can_hoist(&self, arguments: u64, len: u64, count: Option<u64>) -> bool {
        self.blocker(arguments, len, count).is_none()
    }

    /// What keeps a clamp of `[arguments, arguments + len)` (and the
    /// 4-byte `count`) out of the prelude, if anything.
    pub(crate) fn blocker(
        &self,
        arguments: u64,
        len: u64,
        count: Option<u64>,
    ) -> Option<ClampBlock> {
        if self.inline_used {
            return Some(ClampBlock::AfterInline);
        }
        if self.unknown {
            return Some(ClampBlock::Unknown);
        }
        if self.overlaps(arguments, len) || count.is_some_and(|at| self.overlaps(at, 4)) {
            return Some(ClampBlock::Overlap);
        }
        None
    }

    pub(crate) fn clear(&mut self) {
        self.ranges.clear();
        self.unknown = false;
        self.inline_used = false;
    }
}

#[derive(Default)]
struct ArgumentArena {
    chunks: Vec<ArgumentChunk>,
    current: usize,
    used: u64,
    touched: usize,
    oversized: Vec<ID3D12Resource>,
}

impl ArgumentArena {
    fn alloc(
        &mut self,
        host: &Host,
        bytes: u64,
        label: &str,
    ) -> Option<(ID3D12Resource, u64, u64)> {
        let size = bytes.max(16).checked_next_multiple_of(16)?;
        if size > ARGUMENT_CHUNK_MAX {
            let (buffer, _) = host.shared_buffer(size)?;
            name(&buffer, || format!("{label} {size:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.oversized.push(buffer.clone());
            return Some((buffer, 0, gpu));
        }
        while let Some(chunk) = self.chunks.get(self.current) {
            if size <= chunk.capacity - self.used {
                break;
            }
            self.current += 1;
            self.used = 0;
        }
        if self.current == self.chunks.len() {
            let next = self
                .chunks
                .last()
                .map_or(ARGUMENT_CHUNK_FIRST, |c| (2 * c.capacity).min(ARGUMENT_CHUNK_MAX));
            let capacity = size
                .next_multiple_of(ARGUMENT_CHUNK_FIRST)
                .max(next)
                .min(ARGUMENT_CHUNK_MAX);
            let (buffer, _) = host.shared_buffer(capacity)?;
            name(&buffer, || format!("{label} {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.chunks.push(ArgumentChunk {
                buffer,
                gpu,
                capacity,
            });
            self.used = 0;
        }
        let current = self.current;
        let at = self.used;
        let chunk = &self.chunks[current];
        let result = (chunk.buffer.clone(), at, chunk.gpu + at);
        self.used = at + size;
        self.touched = self.touched.max(current + 1);
        Some(result)
    }

    fn touched_buffers(&self) -> Vec<ID3D12Resource> {
        let mut out = Vec::with_capacity(self.touched + self.oversized.len());
        out.extend(self.chunks.iter().take(self.touched).map(|c| c.buffer.clone()));
        out.extend(self.oversized.iter().cloned());
        out
    }

    fn reset(&mut self) -> (Vec<ArgumentChunk>, Vec<ID3D12Resource>) {
        let kept = (self.touched + 1).min(self.chunks.len());
        let spent_chunks = self.chunks.split_off(kept);
        let spent_oversized = std::mem::take(&mut self.oversized);
        self.current = 0;
        self.used = 0;
        self.touched = 0;
        (spent_chunks, spent_oversized)
    }
}

#[derive(Default)]
struct HoistedClamps {
    arena: ArgumentArena,
    pending: Vec<crate::indirect::PendingClamp>,
    budget_gpu: u64,
    hazard: WriteHazard,
    /// The list was started for a clamp that could not run in the prelude
    /// of the one before it (`crate::list::cut_for_clamp`).
    cut: bool,
}

impl HoistedClamps {
    fn touched_buffers(&self) -> Vec<ID3D12Resource> {
        self.arena.touched_buffers()
    }
}

struct ScratchChunk {
    buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer and memory copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy between overlapping title ranges (destination, source, bytes).
    Move(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a copy into it.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings the current buffer view of a table slot up to date: the title
    /// may have rewritten the slot since the draw was recorded.
    Slot(std::sync::Arc<crate::bindless::Viewed>, u32),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then), and
    /// whether their program uses title memory (then the range is withheld
    /// when such a list misses, `crate::memory::withholding`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a title copy intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Raw COM pointers already pushed into [`Self::kept`] by [`HostList::keep`].
    kept_ptrs: std::collections::HashSet<usize, std::hash::BuildHasherDefault<PtrHasher>>,
    /// Pipelines [`HostList::bind_pipeline`] already kept and noted since reset.
    bound_pipelines: std::collections::HashSet<usize>,
    /// Unaligned root-CBV scratch copies (`(source, bytes)` → host GPU address)
    /// valid until the next title copy or list reset.
    filled_scratch: std::collections::HashMap<(usize, usize), u64>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Table slots asked for since the last title write (heap, index).
    slots: std::collections::HashSet<(usize, u32)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Title textures written by GPU shaders that match `VICTORY_TEXTURE_BACK`,
    /// copied into readback buffers when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Recorded GPU-to-CPU texture readbacks to retile into title memory once
    /// the submission completes.
    texture_backs: Vec<crate::texback::Back>,
    /// Last GPU source copied into each aliased host texture in this list.
    texture_aliases: std::collections::HashMap<usize, usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The memory values the list waits for before it starts (a list the
    /// title records with a wait after commands is cut there,
    /// `crate::list`).
    waits: Vec<crate::hold::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Whether the list binds the bitmaps shaders mark their writes through
    /// views in ([`crate::shadow::copy_back_marks`]).
    marks: bool,
    /// Whether `copies` contains any `Step::Heap`, `Step::Texture`, or
    /// `Step::GpuWrite` steps, letting submission prep and completion skip
    /// scanning `copies` for lists that have none.
    has_heaps: bool,
    has_textures: bool,
    has_gpu_writes: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<Box<dyn Fn() + Send>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
}

/// Features reported by the D3D12 device used for rendering. A failed
/// feature query leaves its result unknown.
#[derive(Clone, Copy)]
pub struct MeshSupport {
    pub mesh_shaders: Option<bool>,
    pub shader_model_6_6: Option<bool>,
}

impl MeshSupport {
    /// Translated mesh shaders need both features.
    pub fn supported(self) -> bool {
        self.mesh_shaders == Some(true) && self.shader_model_6_6 == Some(true)
    }
}

/// Checks the active host without loading a title. Returns `None` if
/// VIctory could not initialise its D3D12 host.
pub fn host_mesh_support() -> Option<MeshSupport> {
    Some(host()?.mesh_support())
}

impl Host {
    /// The driver's version, as the system counts it (0: not known).
    ///
    /// Warming keeps this beside the pipelines it made, so that a new driver
    /// makes them stale. Under Wine the number the adapter gives would
    /// describe the Windows-side stub rather than the D3DMetal build that
    /// actually compiles the pipelines, so it answers "not known" until there
    /// is something here worth comparing. That costs a stale warm cache
    /// across a Game Porting Toolkit update: clear the folder by hand after
    /// one.
    pub(crate) fn driver(&self) -> i64 {
        0
    }
}

impl HostGpu for Host {
    type List = HostList;
    type Error = windows::core::Error;
    type Quiet<'a> = Quiet<'a>;

    fn mesh_support(&self) -> MeshSupport {
        static MESH: OnceLock<MeshSupport> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let support = unsafe {
                let mesh_shaders = self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                );
                let shader_model = self.device.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of_val(&model) as u32,
                );
                MeshSupport {
                    mesh_shaders: mesh_shaders
                        .ok()
                        .map(|()| options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED),
                    shader_model_6_6: shader_model
                        .ok()
                        .map(|()| model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0),
                }
            };
            tracing::info!(mesh = support.supported(), "host mesh shader support");
            support
        })
    }

    /// Whether the host's waves have at most 32 lanes, 32 among them
    /// (NVIDIA, Intel): `WaveSize(64)` is refused there, so wave64 compute
    /// programs are translated for waves of 32
    /// (`victory_shader::emit::Options::host_wave32`); logged once.
    fn wave32(&self) -> bool {
        static WAVE32: OnceLock<bool> = OnceLock::new();
        *WAVE32.get_or_init(|| {
            let host = victory_shader::emit::host::Host::from_device(self.device.clone());
            match host.wave32() {
                Ok(wave32) => {
                    tracing::info!(wave32, "host waves of 32 lanes at most");
                    wave32
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host wave lanes not known");
                    false
                }
            }
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            let list7 = self
                .enhanced_barriers()
                .then(|| list.cast::<ID3D12GraphicsCommandList7>().ok())
                .flatten();
            let prelude7 = self
                .enhanced_barriers()
                .then(|| prelude.cast::<ID3D12GraphicsCommandList7>().ok())
                .flatten();
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list7,
                prelude7,
                list,
                epoch: std::sync::atomic::AtomicU64::new(next_epoch()),
                deferred_epoch: std::sync::atomic::AtomicU64::new(next_epoch()),
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                widened_writes: Mutex::default(),
                widened_pending: Mutex::default(),
                widened_sources: Mutex::default(),
                widened_ready: Mutex::default(),
                widened_states: Mutex::default(),
                ours: false.into(),
                memory_writer: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                arguments: Mutex::new(None),
                inline_arguments: Mutex::default(),
                clamps: Mutex::default(),
                marked: std::sync::atomic::AtomicBool::new(false),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                timed: Mutex::default(),
            })
        }
    }

    /// Holds other threads' submissions off until dropped, around the making
    /// of a shader-visible host view heap and its fill with null views
    /// (millions of descriptor copies). Under Wine on D3DMetal, lists other
    /// threads submitted during such a fill lost their views: their
    /// shaders' writes were dropped and their reads gave zeros, whatever
    /// heap the lists bound (VIctory-macOS issue #31). Nothing on a thread
    /// that holds the submission lock already, nor off Wine.
    fn quiet(&self) -> Option<Quiet<'_>> {
        if !crate::under_wine() {
            return None;
        }
        if HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            // A submission being prepared behind one in flight
            // (`VICTORY_PREP_AHEAD=1`) lands it first: the GPU is idle
            // around a heap's fill, as when it is prepared after it.
            flight::land_ahead(flight::Early::Quiet);
            return None;
        }
        let mut held = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let holding = Holding::enter();
        // Ensure the GPU is completely idle before descriptor heap creation /
        // null-fill begins (VIctory-macOS issue #31).
        let landed = self.finish_flight_locked(&mut held, None, None);
        Some(Quiet {
            holding: Some(holding),
            held: Some(held),
            landed,
        })
    }

    /// Runs closed lists and waits for them.
    #[track_caller]
    fn execute(&self, runs: &[&HostList]) {
        self.execute_with_barriers(runs, &[]);
    }
}

impl Host {
    /// Whether the host device reports D3D12 Enhanced Barriers support
    /// (`D3D12_FEATURE_D3D12_OPTIONS12::EnhancedBarriersSupported`).
    pub(crate) fn enhanced_barriers(&self) -> bool {
        self.enhanced_barriers_supported
    }

    /// Runs closed lists and waits for them, inserting a global UAV barrier
    /// command list before each `runs[i]` where `barrier_before.get(i) == Some(&true)`.
    /// Drains any queued async-submit work and any active Flight-1 submission
    /// first so synchronous callers observe strictly ordered GPU state.
    #[track_caller]
    pub(crate) fn execute_with_barriers(&self, runs: &[&HostList], barrier_before: &[bool]) {
        if !HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            crate::queue::before_sync();
        }
        self.execute_sync(runs, barrier_before);
    }

    /// Runs closed lists synchronously under `self.submitted`, finishing any
    /// active Flight-1 submission first.
    pub(crate) fn execute_sync(&self, runs: &[&HostList], barrier_before: &[bool]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.finish_flight_locked(&mut submitted, None, None);

        #[cfg(feature = "diagnostics")]
        let gpu_timing = timing();
        #[cfg(not(feature = "diagnostics"))]
        let gpu_timing = false;

        let mut prep = self.prepare_submission(*submitted, runs, barrier_before, gpu_timing);
        let run_refs: Vec<&HostList> = prep
            .uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &run_refs[..];

        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if gpu_timing {
            prep.firsts
                .iter()
                .zip(prep.firsts.iter().skip(1).chain([&prep.lists.len()]))
                .map(|(&a, &b)| &prep.lists[a..b])
                .collect()
        } else {
            vec![&prep.lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&prep.lists[..]];
        let mut removed_in = None;
        let mark = self.mark();
        // A failed earlier call may leave the queue able to run the final mark.
        // Give each native call its own mark so failure recovery can see it.
        if mark.is_none() {
            prep.native_boundaries.clear();
        }
        let mut split_pool = (!prep.native_boundaries.is_empty())
            .then(|| self.split_marks.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(pool) = split_pool.as_deref_mut() {
            if let Some(marks) = pool.as_mut()
                && let Err(error) = (marks.len()..prep.native_boundaries.len()).try_for_each(|_| {
                    marks.push(Mark::new(self)?);
                    Ok::<_, windows::core::Error>(())
                })
            {
                tracing::warn!(
                    %error,
                    "could not make split submission marks, keeping the native batch together"
                );
                *pool = None;
            }
            if pool.is_none() {
                prep.native_boundaries.clear();
            }
        }
        let split_marks = split_pool
            .as_deref()
            .and_then(Option::as_deref)
            .unwrap_or(&[]);
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            SUBMISSIONS_BEGUN.store(*submitted, Ordering::Release);
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
            // Last, the list that shows the submission ran.
            let mut group = group.to_vec();
            if let Some(mark) = mark {
                mark.arm(*submitted);
                group.push(mark.list());
            }
            let queue = self.queue();
            // SAFETY: closed lists; the fence and event are ours.
            let waited = unsafe {
                // All native calls share the existing fence signal and wait.
                let mut first = 0;
                for (&end, mark) in prep.native_boundaries.iter().zip(split_marks) {
                    mark.arm(*submitted);
                    let mut marked = group[first..end].to_vec();
                    marked.push(mark.list());
                    #[cfg(test)]
                    tests::native_call(marked.len());
                    queue.ExecuteCommandLists(&marked);
                    first = end;
                }
                #[cfg(test)]
                tests::native_call(group.len() - first);
                queue.ExecuteCommandLists(&group[first..]);
                let waited = queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        self.wait_for(*submitted);
                        true
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "host submission not waited for");
                        false
                    }
                }
            };
            #[cfg(test)]
            tests::native_call_group(waited);
            if waited
                && let Some(mark) = mark
                && (!mark.saw(*submitted)
                    || split_marks
                        .iter()
                        .take(prep.native_boundaries.len())
                        .any(|m| !m.saw(*submitted)))
                // SAFETY: a plain query.
                && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
            {
                self.lost(&mut submitted, mark);
            }
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 {
                // SAFETY: a plain query.
                let frequency = unsafe { queue.GetTimestampFrequency() }.unwrap_or(1);
                gpu_times(
                    started.elapsed(),
                    &runs[i].slowest(frequency, usize::MAX).unwrap_or_default(),
                );
            }
            #[cfg(feature = "diagnostics")]
            if gpu_timing && timing_cuts_on() {
                let timed = runs[i].timed.lock().unwrap_or_else(|e| e.into_inner());
                GPU_WALLS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert_with(Default::default)
                    .add(
                        started.elapsed().as_secs_f64() * 1e3,
                        timed.as_ref(),
                        crate::queue::presented(),
                    );
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i]
                    .notes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .len();
                tracing::warn!(
                    ms = started.elapsed().as_millis(),
                    commands,
                    "slow host list"
                );
                match runs[i].slowest(frequency, 6) {
                    Some(slowest) => {
                        for (ms, op) in slowest {
                            tracing::warn!("slow list command, {ms:.1} ms: {op}");
                        }
                    }
                    None => {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            tracing::warn!(
                                frequency,
                                "GPU timestamps do not advance on this adapter: \
                                 slow lists are timed whole, their commands are not"
                            )
                        });
                    }
                }
            }
            if groups.len() > 1 && removed_in.is_none() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    removed_in = Some(i);
                }
            }
            let _ = (i, started);
        }
        drop(split_pool);
        if flight1_enabled() {
            self.note_gpu_free(std::time::Instant::now());
        }
        let done = *submitted;
        SUBMISSIONS_DONE.store(done, Ordering::Release);
        // The GPU fence has completed: return worker-owned staging chunks now.
        if let Some(u) = &prep.uploads {
            u.release_worker();
        }
        self.finish_post_gpu(
            runs,
            prep.batch,
            prep.heaps,
            prep.started,
            prep.prepared,
            prep.prepared.elapsed(),
            removed_in,
            done,
        );
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if prep.traced {
            let sub = *submitted;
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            drop(run_refs);
            drop(prep.uploads);
            drop(submitted);
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// Prepares a submission under `self.submitted` once any prior flight has
    /// finished: evicts unused textures, syncs descriptor heaps, trims idle
    /// lists, records texture uploads, inserts coalesced UAV barriers, runs
    /// deferred CPU copies/refreshes, and builds preludes (including hoisted
    /// indirect clamps).
    fn prepare_submission(
        &self,
        submitted: u64,
        runs: &[&HostList],
        barrier_before: &[bool],
        gpu_timing: bool,
    ) -> PreparedSubmission {
        let started = std::time::Instant::now();
        let mut clock = PhaseClock(started);
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ExecuteLists, runs.len() as u64);
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first: the last submission has run.
        // With `VICTORY_PREP_AHEAD=1` a submission still in flight lands
        // first (nothing otherwise): eviction gives up what it may use.
        if crate::resource::evict_due() {
            clock.lap(Phase::Other);
            flight::land_ahead(flight::Early::Eviction);
            clock.lap(Phase::LandingEarly);
        }
        crate::resource::evict_unused();
        // The first shader writing through a heap may run in this
        // submission: unordered views left null are made again first.
        if crate::bindless::take_remake() {
            clock.lap(Phase::Evicting);
            flight::land_ahead(flight::Early::Remake);
            clock.lap(Phase::LandingEarly);
            crate::descriptor::remake_unordered();
        }
        clock.lap(Phase::Evicting);
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        crate::descriptor::sync_heaps();
        // Slots left as they were while the last submission ran, because
        // the texture they now view was made meanwhile (`VICTORY_SHOW_AFTER`).
        crate::descriptor::show_held();
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        clock.lap(Phase::HeapSync);
        // Textures heap-indexing shaders may read, and textures read through
        // tables, uploaded first by a list of ours where their title memory
        // changed.
        let mut heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = Vec::new();
        let mut textures: Vec<(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
        )> = Vec::new();
        for list in runs {
            let deferred = list.deferred();
            if !deferred.has_heaps && !deferred.has_textures {
                continue;
            }
            for step in &deferred.copies {
                match step {
                    Step::Heap(v) => heaps.push(v.clone()),
                    Step::Texture(staging, copy) => textures.push((staging.clone(), copy.clone())),
                    _ => {}
                }
            }
        }
        // Every submission, not only those with textures to bring up: the
        // staging pools and the upload list (it holds what the last
        // submission staged) age on each one (`uploads`), and so do the
        // idle spare lists. The last submission has run.
        crate::list::trim_idle();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = crate::bindless::uploads(&heaps, &textures);
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        clock.lap(Phase::Uploads);
        let uploads_offset = usize::from(uploads.is_some());
        let run_refs: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &run_refs[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        let split_preludes = !gpu_timing && split_uav_srv_prelude();
        let needed_barriers = barrier_before.iter().copied().filter(|&b| b).count();
        // The barrier lists are shared: one submission runs them at a time.
        if needed_barriers > 0 {
            clock.lap(Phase::Other);
            flight::land_ahead(flight::Early::Barriers);
            clock.lap(Phase::LandingEarly);
        }
        let mut barrier_pool = (needed_barriers > 0)
            .then(|| self.barrier_lists.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(pool) = barrier_pool.as_deref_mut()
            && let Some(barriers) = pool.as_mut()
            && let Err(error) = (barriers.len()..needed_barriers).try_for_each(|_| {
                barriers.push(BarrierList::new(self)?);
                Ok::<_, windows::core::Error>(())
            })
        {
            tracing::warn!(
                %error,
                "could not make coalesced UAV barrier lists, submitting without extra barriers"
            );
            *pool = None;
        }
        let barrier_lists = barrier_pool
            .as_deref()
            .and_then(Option::as_deref)
            .unwrap_or(&[]);
        let mut barrier_cursor = 0usize;
        let mut native_boundaries = Vec::new();
        let mut lists: Vec<Option<ID3D12CommandList>> =
            Vec::with_capacity(2 * runs.len() + needed_barriers);
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        // Title copies are done here on the CPU, ahead of the GPU work before
        // them: a submission in flight that reads their destinations' host
        // copies lands first (`VICTORY_PREP_AHEAD=1`).
        if runs.iter().any(|r| r.batch_flags().has_title_copy) {
            clock.lap(Phase::Other);
            flight::land_ahead(flight::Early::TitleCopy);
            clock.lap(Phase::LandingEarly);
        }
        let mut batch = Batch::take();
        clock.lap(Phase::Other);
        for (idx, list) in runs.iter().enumerate() {
            list.run_deferred(&mut batch);
            clock.lap(Phase::ListSteps);
            firsts.push(lists.len());
            let needs_barrier = idx
                .checked_sub(uploads_offset)
                .and_then(|i| barrier_before.get(i).copied())
                .unwrap_or(false);
            if needs_barrier
                && !lists.is_empty()
                && let Some(barrier) = barrier_lists.get(barrier_cursor)
            {
                lists.push(barrier.list());
                barrier_cursor += 1;
            }
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some((prelude, uav_to_srv)) = list.prelude(split_preludes) {
                if uav_to_srv && !lists.is_empty() {
                    native_boundaries.push(lists.len());
                }
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
            clock.lap(Phase::Preludes);
        }
        drop(barrier_pool);
        batch.flush();
        clock.lap(Phase::Refreshes);
        #[cfg(feature = "diagnostics")]
        batch.log_refreshes(submitted + 1, runs.len());
        #[cfg(not(feature = "diagnostics"))]
        let _ = submitted;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::FaultsPrep,
            crate::copystat::page_faults().saturating_sub(faults),
        );
        if !native_boundaries.is_empty() {
            static FIRST_SPLIT: std::sync::Once = std::sync::Once::new();
            FIRST_SPLIT.call_once(|| {
                tracing::info!(
                    native_calls = native_boundaries.len() + 1,
                    "split host submission at deferred UAV-to-SRV preludes"
                );
            });
        }
        // What a hang would be waiting on (`Host::wait_for` names it after 8 s).
        {
            let mut waiting = WAITING_ON.lock().unwrap_or_else(|e| e.into_inner());
            waiting.0 = runs.len();
            waiting.1.clear();
            waiting.1.extend(
                runs.iter()
                    .map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            );
        }
        drop(run_refs);
        #[cfg(not(feature = "diagnostics"))]
        let _ = firsts;
        // The hang note is part of the preparing, which it used to follow.
        clock.lap(Phase::Other);
        let prepared = clock.0;
        PreparedSubmission {
            lists,
            #[cfg(feature = "diagnostics")]
            firsts,
            native_boundaries,
            batch,
            heaps,
            uploads,
            started,
            prepared,
            #[cfg(feature = "diagnostics")]
            traced,
        }
    }

    /// Runs post-GPU completion work under `self.submitted` after the host
    /// fence has completed: timing/removal checks, mark & shadow copy-back,
    /// texture readback landings, `after` callbacks, bottom-of-pipe writes,
    /// deferred retirement, and info-queue logs.
    #[allow(clippy::too_many_arguments)]
    fn finish_post_gpu(
        &self,
        runs: &[&HostList],
        batch: Batch,
        heaps: Vec<std::sync::Arc<crate::bindless::Viewed>>,
        started: std::time::Instant,
        prepared: std::time::Instant,
        gpu: std::time::Duration,
        removed_in: Option<usize>,
        done: u64,
    ) {
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines = pipeline_names(
                runs.iter()
                    .map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            );
            tracing::warn!(
                ms = gpu.as_millis(),
                lists = runs.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, removed_in)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> =
            std::collections::HashMap::new();
        // Shaders that read memory with no copy computed something else than
        // the console's: it stays out of title memory, where it broke the
        // title's own data (runs c1, c4), and so does what every program
        // using title memory wrote while any misses
        // (`crate::memory::withholding`). Only lists with such programs have
        // checks; all are asked before anything goes back. Other programs in
        // those lists write through bounded views from their own inputs:
        // their ranges go back (the imposter transforms beside the BVH
        // builder in one list, run light13).
        let missed: Vec<bool> = runs
            .iter()
            .map(|list| list.deferred().missed.iter().any(|check| check()))
            .collect();
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        for (list, &missed) in runs.iter().zip(&missed) {
            let deferred = list.deferred();
            if !deferred.has_gpu_writes {
                continue;
            }
            let missed = missed || withholding && !deferred.missed.is_empty();
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _, memory) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    // A range both kinds declare is withheld whole.
                    let goes_back = !(missed && *memory);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= goes_back,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, goes_back));
                        }
                    }
                }
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held
                .iter()
                .any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        flight::probe_landing_starts();
        for (_, address, len, _) in &back {
            flight::probe_wrote(*address, *len);
        }
        crate::shadow::copy_back_marks(&withheld, withholding);
        // Texture views left null that shaders read: made for what follows.
        let read = crate::shadow::take_reads();
        if !read.is_empty() {
            crate::descriptor::make_read(&read);
        }
        // Lanes that gave up waiting for a 64-bit atomic's lock: their
        // updates may be lost (`victory_shader::emit`, "Atomic locks").
        let gave_up = crate::shadow::take_gave_up();
        if gave_up > 0 {
            let before = GAVE_UP.fetch_add(gave_up, Ordering::Relaxed);
            let total = before + gave_up;
            if before == 0 || before.ilog2() != total.ilog2() {
                tracing::warn!(
                    gave_up,
                    total,
                    "lanes gave up waiting for a 64-bit atomic's lock and went on without it, so their updates may be lost (#240)"
                );
            }
        }
        let total_back_bytes: usize = back.iter().filter(|b| b.3).map(|b| b.2).sum();
        if crate::shadow::parallel_copies_enabled()
            && total_back_bytes >= crate::shadow::PARALLEL_COPY_BACK_MIN_BYTES
        {
            type BackGroup = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize, bool)>);
            let mut groups: Vec<BackGroup> = Vec::new();
            for (shadow, address, len, goes_back) in &back {
                match groups
                    .iter_mut()
                    .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
                {
                    Some((_, items)) => items.push((*address, *len, *goes_back)),
                    None => groups.push((shadow.clone(), vec![(*address, *len, *goes_back)])),
                }
            }
            let shadow_ranges: Vec<(u64, u64)> = groups.iter().map(|(s, _)| s.range()).collect();
            if groups.len() >= 2 && crate::shadow::copy_back_ranges_disjoint(&shadow_ranges) {
                crate::texture::in_parallel(groups.len(), |i| {
                    let (shadow, items) = &groups[i];
                    for &(address, len, goes_back) in items {
                        if goes_back {
                            let written = shadow.copy_back(address, len);
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        } else {
                            shadow.keep_writes(address, len);
                            crate::memory::note_withheld(len as u64);
                            crate::memory::drop_withheld(address, len);
                        }
                    }
                });
            } else {
                for (shadow, address, len, goes_back) in &back {
                    let (address, len) = (*address, *len);
                    if *goes_back {
                        let written = shadow.copy_back(address, len);
                        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                    } else {
                        shadow.keep_writes(address, len);
                        crate::memory::note_withheld(len as u64);
                        crate::memory::drop_withheld(address, len);
                    }
                }
            }
        } else {
            for (shadow, address, len, goes_back) in &back {
                let (address, len) = (*address, *len);
                if *goes_back {
                    let written = shadow.copy_back(address, len);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                } else {
                    shadow.keep_writes(address, len);
                    crate::memory::note_withheld(len as u64);
                    crate::memory::drop_withheld(address, len);
                }
            }
        }
        for list in runs {
            for back in &list.deferred().texture_backs {
                crate::texback::land(back);
            }
        }
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::FaultsBack,
            crate::copystat::page_faults().saturating_sub(faults),
        );
        for list in runs {
            for &(at, _, bytes) in &list.deferred().writes {
                flight::probe_wrote(at, bytes);
            }
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        // The submission has run and the next waits for our lock. Host
        // copies and retired GPU objects stay alive across `bury_lag`
        // submissions (~2 frames by default) so D3DMetal/AGX never faults on
        // recently retired buffer or texture virtual addresses.
        retire_any((batch, back, withheld, heaps));
        advance_retired(done);
        self.log_messages();
    }

    /// Waits until the host fence reaches `value`. The event is only a
    /// wake-up: the fence's completed value decides. Under D3DMetal a
    /// submission's work can finish without its event ever being set
    /// (#50: the title's main thread slept for good in this wait while the
    /// queue was idle), so the wait wakes every second to check the fence
    /// itself. A late event left set by an earlier wait cannot end a later
    /// wait early, as the fence is checked after every wake-up too.
    fn wait_for(&self, value: u64) {
        self.wait_for_on_event(self.event, value, 1000);
    }

    /// Waits until the host fence reaches `value`, polling `event` every
    /// `poll_ms` milliseconds so a dropped D3DMetal event (#50) never hangs.
    pub(super) fn wait_for_on_event(&self, event: HANDLE, value: u64, poll_ms: u32) {
        use std::sync::atomic::AtomicU64;
        static MISSED: AtomicU64 = AtomicU64::new(0);
        let started = std::time::Instant::now();
        let mut next_log_secs = 1u64;
        let mut logged_hang = false;
        loop {
            // SAFETY: our fence.
            if unsafe { self.fence.GetCompletedValue() } >= value {
                return;
            }
            // SAFETY: our event and fence.
            let woke = unsafe { WaitForSingleObject(event, poll_ms) };
            // SAFETY: as above.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed >= value {
                if woke != windows::Win32::Foundation::WAIT_OBJECT_0
                    && started.elapsed().as_millis() >= 1000
                {
                    let n = MISSED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        tracing::warn!(n, value, completed, "a host submission finished without setting its event; went on after checking the fence");
                    }
                }
                return;
            }
            if woke != windows::Win32::Foundation::WAIT_OBJECT_0 {
                let seconds = started.elapsed().as_secs();
                if seconds >= next_log_secs {
                    tracing::warn!(seconds, value, completed, "still waiting for the host GPU");
                    next_log_secs = (next_log_secs * 2).max(seconds + 1);
                }
                if !logged_hang && seconds >= 8 {
                    logged_hang = true;
                    let what = {
                        let waiting = WAITING_ON.lock().unwrap_or_else(|e| e.into_inner());
                        let pipelines = pipeline_names(waiting.1.iter().cloned());
                        format!("{} lists, pipelines {pipelines:?}", waiting.0)
                    };
                    tracing::error!(%what, value, "host GPU hang: the submission waited on");
                }
            }
        }
    }

    /// The queue submissions go to now.
    pub(crate) fn queue(&self) -> ID3D12CommandQueue {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The list submissions end with ([`Mark`]); `None` if it could not be
    /// made, and then submissions are not checked.
    /// The mark the host submission with fence `value` ends with: [`Self::mark`],
    /// or with `VICTORY_PREP_AHEAD=1` the second one for even values (two
    /// submissions in flight have consecutive values).
    fn mark_for(&self, value: u64) -> Option<&Mark> {
        if flight::prep_ahead_enabled() && value.is_multiple_of(2) {
            return self
                .mark_even
                .get_or_init(|| {
                    Mark::new(self)
                        .inspect_err(|e| tracing::warn!(error = %e, "no second submission mark"))
                        .ok()
                })
                .as_ref();
        }
        self.mark()
    }

    fn mark(&self) -> Option<&Mark> {
        self.mark
            .get_or_init(|| {
                Mark::new(self)
                    .inspect_err(|e| tracing::warn!(error = %e, "no submission mark: submissions the host GPU drops go unseen"))
                    .ok()
            })
            .as_ref()
    }

    /// After a submission that did not run to its mark: what it drew is
    /// missing. If the queue no longer runs work, a new queue replaces it,
    /// and the next present makes its swap chain on that one. Under the
    /// submission lock.
    fn lost(&self, submitted: &mut u64, mark: &Mark) {
        let lost = LOST[0].fetch_add(1, Ordering::Relaxed) + 1;
        // A lane that held a lock of an emulated 64-bit atomic when the GPU
        // ended its shader never frees it (`victory_shader::emit`, "Atomic
        // locks"): free them all before anything else runs.
        let freed = crate::shadow::free_atomic_locks();
        if freed > 0 {
            tracing::warn!(
                freed,
                "atomic locks left taken by the ended submission freed"
            );
        }
        if self.runs(submitted, mark) {
            if lost.is_power_of_two() {
                tracing::warn!(
                    lost,
                    "the host GPU ended a submission before it finished, and what it drew is missing \
                     (under D3DMetal, macOS ends a command buffer when the GPU locks up, as on a draw \
                     that holds it too long, or takes a page fault)"
                );
            }
            return;
        }
        if let Err(e) = self.replace_queue() {
            fail(format_args!(
                "the host queue stopped running work, and no new queue could be made: {e}"
            ));
        }
        if !self.runs(submitted, mark) {
            fail(format_args!(
                "the host queue stopped running work, and a new queue does not run it either"
            ));
        }
        let queues = LOST[1].fetch_add(1, Ordering::Relaxed) + 1;
        if queues.is_power_of_two() {
            tracing::warn!(
                lost,
                queues,
                "the host queue stopped running work after the GPU ended a submission \
                 (macOS ignores what is submitted to it): moved to a new queue"
            );
        }
    }

    /// Whether the queue runs work: the mark alone, as a submission of its
    /// own. Under the submission lock.
    fn runs(&self, submitted: &mut u64, mark: &Mark) -> bool {
        *submitted += 1;
        SUBMISSIONS_BEGUN.store(*submitted, Ordering::Release);
        mark.arm(*submitted);
        let queue = self.queue();
        // SAFETY: a closed list; the fence and event are ours.
        unsafe {
            queue.ExecuteCommandLists(&[mark.list()]);
            let waited = queue
                .Signal(&self.fence, *submitted)
                .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
            match waited {
                Ok(()) => self.wait_for(*submitted),
                Err(e) => tracing::warn!(error = %e, "host check not waited for"),
            }
        }
        mark.saw(*submitted)
    }

    /// A new direct queue in place of the current one. Under the submission
    /// lock, so no submission is on the old queue.
    fn replace_queue(&self) -> windows::core::Result<()> {
        // SAFETY: plain object creation.
        let queue: ID3D12CommandQueue = unsafe {
            self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }?;
        *self.queue.lock().unwrap_or_else(|e| e.into_inner()) = queue;
        Ok(())
    }


    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC {
                                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                Count: STAMPS,
                                NodeMask: 0,
                            },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran);
    /// `None` when its timestamps did not advance (D3DMetal reads them all
    /// as 0, at a frequency of 60).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Option<Vec<(f64, String)>> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Some(Vec::new());
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (count + 1) * 8,
        };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Some(Vec::new());
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        if count > 0 && stamps.windows(2).all(|w| w[0] == w[1]) {
            return None;
        }
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (
                    ticks as f64 * 1e3 / frequency.max(1) as f64,
                    notes[i].0.chars().take(300).collect(),
                )
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        Some(times)
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count =
                (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list
                    .ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        self.textures_back();
        self.widen_close();
        self.hoisted_clamps_close();
        self.end_marker();
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        let texture_backs = {
            let mut deferred = self.deferred();
            let backs = std::mem::take(&mut deferred.texture_backs);
            *deferred = Deferred::default();
            backs
        };
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_writes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        pending.events.clear();
        pending.writes.clear();
        drop(pending);
        self.widened_sources.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.widened_states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            // The hang note may still hold the set: a new one, not a copy.
            reset_pipelines(&mut self.pipelines.lock().unwrap_or_else(|e| e.into_inner()));
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.timed.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.trim_scratch();
        self.reset_hoisted_clamps();
        self.reset_inline_arguments();
        self.marked.store(false, Ordering::Relaxed);
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        // Recycle readback buffers only once the command allocator and list
        // have reset cleanly so no live command list still references them.
        for back in texture_backs {
            back.recycle();
        }
        self.changed_deferred();
    }

    /// Bumps the list's texture/state mutation epoch whenever a recorded
    /// transition or twin copy/write changes what subsequent texture
    /// descriptor preparation would need to do.
    pub(crate) fn changed_something(&self) {
        self.epoch.store(next_epoch(), Ordering::Relaxed);
    }

    /// Bumps both the deferred-dedup epoch and texture/state epoch when
    /// deferred refresh/slot sets are cleared (`reset` or title copy).
    fn changed_deferred(&self) {
        let next = next_epoch();
        self.deferred_epoch.store(next, Ordering::Relaxed);
        self.epoch.store(next, Ordering::Relaxed);
    }

    /// Monotonic epoch of the last texture/state or reset mutation on this list.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Monotonic epoch of the last deferred refresh/slot invalidation on this list.
    pub(crate) fn deferred_epoch(&self) -> u64 {
        self.deferred_epoch.load(Ordering::Relaxed)
    }

    /// Number of deferred submission-time steps currently queued on this list.
    pub(crate) fn deferred_steps(&self) -> usize {
        self.deferred().copies.len()
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(ID3D12Resource, u32)> {
        // One host texture for the whole move (`crate::resource::evict` may
        // give it up meanwhile; this list then holds it).
        let copy = crate::resource::host_copy(resource)?;
        // Unordered access through the twin skips `track` below.
        copy.note_state(wanted);
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = copy.legal_state(wanted);
        // Cloned (two calls into the host) only for a barrier.
        self.track(&copy, wanted)
            .map(|before| (copy.texture.clone(), before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        copy.note_state(wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                if !self.ours.load(Ordering::Relaxed) {
                    copy.stamp();
                }
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    self.changed_something();
                    Some(before)
                } else {
                    None
                }
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = copy.legal_state(wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
        whole: bool,
        stores: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                let u = v.insert(crate::twin::Use::new(depth, side, writes, whole));
                if stores {
                    u.store();
                }
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let copy = o.get_mut().again(side, writes, whole);
                if stores {
                    o.get_mut().store();
                }
                if copy.is_some() || writes {
                    self.changed_something();
                }
                copy
            }
        }
    }

    /// The list is one of ours (uploads): what it uses is no use of the
    /// title's.
    pub(crate) fn set_ours(&self) {
        self.ours.store(true, Ordering::Relaxed);
    }

    /// What the list's shaders write from now until the guard goes is
    /// written by a program using title memory (`uses`) or not.
    pub(crate) fn writer_uses_memory(&self, uses: bool) -> impl Drop + '_ {
        struct Reset<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        self.memory_writer.store(uses, Ordering::Relaxed);
        Reset(&self.memory_writer)
    }

    /// Marks title texture `resource` as GPU-written on this command list so
    /// eligible 2D textures (`VICTORY_TEXTURE_BACK`) are copied into a readback
    /// buffer when the list closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred
            .back_wanted
            .iter()
            .any(|target| target.owner() == resource as usize)
            && let Some(target) = crate::texback::Target::new(resource)
        {
            deferred.back_wanted.push(target);
        }
    }

    /// Whether this command list has recorded GPU-to-CPU texture readbacks.
    #[allow(dead_code)]
    pub(crate) fn has_texture_backs(&self) -> bool {
        !self.deferred().texture_backs.is_empty()
    }

    /// Destination title-memory ranges `[start, end)` updated when this command
    /// list's GPU-to-CPU texture readbacks land.
    #[allow(dead_code)]
    pub(crate) fn texture_back_ranges(&self) -> Vec<(usize, usize)> {
        self.deferred()
            .texture_backs
            .iter()
            .map(crate::texback::Back::range)
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn texture_back_raw_ptrs_for_test(&self) -> Vec<usize> {
        self.deferred()
            .texture_backs
            .iter()
            .map(crate::texback::Back::raw_resource_ptr)
            .collect()
    }

    /// Before the list closes: copies eligible GPU-written textures into host
    /// readback buffers (`crate::texback::record`).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            if !target.is_valid() {
                continue;
            }
            if let Some(barrier) = self.move_texture(&target.copy, COPY_SOURCE) {
                // SAFETY: open command list and valid resource transition.
                unsafe { self.list.ResourceBarrier(&[barrier]) };
            }
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn keep_upload_for_test(&self, upload: ID3D12Resource) {
        self.deferred().uploads.push(upload);
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// A widened SRV is about to be read by this list. A view first made after
    /// an earlier barrier also needs the writes retained since this list began.
    pub(crate) fn mark_widened_read(&self, buffer: &ID3D12Resource) {
        let key = buffer.as_raw() as usize;
        let ready = self
            .widened_ready
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&key);
        if !ready {
            self.sync_widened();
        }
        // Reads need no transition: a cached view is either still common or
        // was left shader-readable by its last widening copy.
    }

    /// A UAV wrote a shadow range that may feed a widened SRV after a later
    /// buffer barrier. Keep writes even before the first widened view exists.
    pub(crate) fn mark_widened_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        if !crate::under_wine() || len == 0 {
            return;
        }
        self.widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .events
            .push(WidenedEvent::Write((shadow.clone(), address, len)));
    }

    /// A heap-indexing shader may write through any of this heap's current
    /// UAV views. Its ranges are read at the barrier, not while binding.
    pub(crate) fn mark_widened_heap(&self, heap: &std::sync::Arc<crate::bindless::Viewed>) {
        if !crate::under_wine() {
            return;
        }
        self.widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .events
            .push(WidenedEvent::Heap(heap.clone()));
    }

    /// Keeps an exact source shadow's arena slice alive until the list resets.
    pub(crate) fn keep_shadow(&self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let mut sources = self.widened_sources.lock().unwrap_or_else(|e| e.into_inner());
        if !sources.iter().any(|other| std::sync::Arc::ptr_eq(shadow, other)) {
            sources.push(shadow.clone());
        }
    }

    /// After the title's buffer/UAV barrier, widen the shadow ranges written
    /// before it so later SRVs in this submission see the UAV output.
    pub(crate) fn sync_widened(&self) {
        let mut pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let WidenedPending { events, writes } = &mut *pending;
        // A heap event becomes the UAV ranges the heap's views reach now.
        // Every event of a heap expands to the same ranges (read now), and
        // the last one's copies come after, so over, the earlier ones'.
        let last = last_heap_events(events);
        for (i, event) in events.drain(..).enumerate() {
            match event {
                WidenedEvent::Write(write) => writes.push(write),
                WidenedEvent::Heap(heap) => {
                    if last.get(&(std::sync::Arc::as_ptr(&heap) as usize)) == Some(&i) {
                        if let Some(cached) = crate::bindless::unordered_writes(&heap) {
                            writes.extend(cached.iter().cloned());
                        } else {
                            crate::bindless::widened_writes_into(&heap, writes);
                        }
                    }
                }
            }
        }
        if !crate::shadow::WIDENED_ANY.load(Ordering::Acquire) {
            self.widened_writes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .append(writes);
            return;
        }
        let mut ready = self.widened_ready.lock().unwrap_or_else(|e| e.into_inner());
        if writes.is_empty()
            && crate::shadow::widened_all()
                .iter()
                .all(|view| ready.contains(&(view.buffer.as_raw() as usize)))
        {
            return;
        }
        let mut history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if history.is_empty() && writes.is_empty() {
            return;
        }
        // Writes a failed widening leaves stay for the next barrier.
        let Some(views) = crate::widened::sync(self, &history, writes, &ready) else {
            return;
        };
        history.append(writes);
        ready.extend(views);
    }

    /// The widened history (address, bytes per write), where the history
    /// and the buffer of writes not widened yet are, and whether anything
    /// waits for a barrier (tests).
    #[cfg(test)]
    pub(crate) fn widened_state(&self) -> (Vec<(usize, usize)>, *const u8, *const u8, bool) {
        let history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        (
            history.iter().map(|&(_, at, len)| (at, len)).collect(),
            history.as_ptr().cast(),
            pending.writes.as_ptr().cast(),
            !pending.events.is_empty() || !pending.writes.is_empty(),
        )
    }

    /// Moves widened buffers into the unordered state for widening copies.
    pub(crate) fn widen_begin_all(&self, buffers: &[ID3D12Resource]) {
        for buffer in buffers {
            self.widen_state(buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    /// Widening copies are complete; later shaders read these buffers.
    pub(crate) fn widen_end_all(&self, buffers: &[ID3D12Resource]) {
        for buffer in buffers {
            self.widen_state(buffer, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }
    }

    fn widen_state(&self, buffer: &ID3D12Resource, wanted: D3D12_RESOURCE_STATES) {
        let key = buffer.as_raw() as usize;
        let mut states = self.widened_states.lock().unwrap_or_else(|e| e.into_inner());
        let before = states.get(&key).map_or(D3D12_RESOURCE_STATE_COMMON, |(_, state)| *state);
        if before == wanted {
            return;
        }
        states.insert(key, (buffer.clone(), wanted));
        drop(states);
        self.keep(buffer);
        // SAFETY: an open list and a live widened buffer.
        unsafe {
            self.list.ResourceBarrier(&[transition(buffer, before, wanted)]);
        }
    }

    /// Returns widened buffers to common so the CPU can refresh them before
    /// the next submission.
    fn widen_close(&self) {
        let mut history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let used = history.len();
        empty_widened(&mut history, used);
        drop(history);
        let mut pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        pending.events.clear();
        empty_widened(&mut pending.writes, used);
        drop(pending);
        self.widened_ready.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let states = std::mem::take(&mut *self.widened_states.lock().unwrap_or_else(|e| e.into_inner()));
        for (_, (buffer, before)) in states {
            if before != D3D12_RESOURCE_STATE_COMMON {
                // SAFETY: an open list and a live widened buffer.
                unsafe {
                    self.list.ResourceBarrier(&[transition(&buffer, before, D3D12_RESOURCE_STATE_COMMON)]);
                }
            }
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self, detect_uav_to_srv: bool) -> Option<(ID3D12CommandList, bool)> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies start from the real states, then the list's moves from
        // them; only after both do they become the states the list leaves.
        let mut uav_to_srv = false;
        let mut moves = || -> Vec<D3D12_RESOURCE_BARRIER> {
            states
                .values()
                .filter_map(|(first, _, copy)| {
                    let real = copy.state.load(Ordering::Relaxed);
                    (real != *first).then(|| {
                        uav_to_srv |= detect_uav_to_srv
                            && real == D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32
                            && *first & D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32 != 0;
                        transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(*first as i32))
                    })
                })
                .collect()
        };
        let left = || {
            for (_, last, copy) in states.values() {
                copy.state.store(*last, Ordering::Relaxed);
            }
        };
        let (clamp_budget, clamp_buffers, pending_clamps) = {
            let clamps = self.clamps.lock().unwrap_or_else(|e| e.into_inner());
            if clamps.pending.is_empty() {
                (0, Vec::new(), Vec::new())
            } else {
                (clamps.budget_gpu, clamps.touched_buffers(), clamps.pending.clone())
            }
        };
        if copies.is_empty()
            && pending_clamps.is_empty()
            && states.values().all(|(first, _, copy)| copy.state.load(Ordering::Relaxed) == *first)
        {
            left();
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = t.legal_state(wanted);
                        (real != wanted).then(|| {
                            let there = transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32));
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            there
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                let barriers = moves();
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if !pending_clamps.is_empty() {
                    crate::indirect::record_batch(
                        list,
                        self.narrow_clamp_sync_prelude(),
                        clamp_budget,
                        &clamp_buffers,
                        &pending_clamps,
                    );
                }
                list.Close()
            })
        };
        left();
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok().map(|list| (list, uav_to_srv))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps()
                && at < STAMPS - 1
            {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list
                    .SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// With `VICTORY_GPU_TIMING_CUTS`: whether a draw or dispatch of `key`
    /// (`crate::timing_cuts::key`) belongs in a host list after this one,
    /// which runs another kind or pipeline.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn timed_cuts(&self, (key, compute): (usize, bool)) -> bool {
        let timed = self.timed.lock().unwrap_or_else(|e| e.into_inner());
        crate::timing_cuts::cuts(timed.as_ref(), key, compute)
    }

    /// Counts a draw or dispatch of `key` about to be recorded into the
    /// list, whose kind and pipeline it gives the list if it is the first
    /// (`name`: the pipeline's shaders).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn timed_command(
        &self,
        (key, compute): (usize, bool),
        name: impl FnOnce() -> String,
    ) {
        let mut timed = self.timed.lock().unwrap_or_else(|e| e.into_inner());
        match timed.as_mut() {
            Some(t) => t.commands += 1,
            None => {
                *timed = Some(crate::timing_cuts::Timed {
                    key,
                    compute,
                    commands: 1,
                    name: name(),
                })
            }
        }
    }

    /// Takes back the last [`HostList::timed_command`]: its command went
    /// into the next host list after all (a cut for an indirect clamp).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn timed_command_moved(&self) {
        let mut timed = self.timed.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = timed.as_mut() {
            t.commands = t.commands.saturating_sub(1);
            if t.commands == 0 {
                *timed = None;
            }
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            // Held by the hang note only between a submission and the
            // list's reset, which gives the list a set of its own.
            std::sync::Arc::make_mut(&mut pipelines).insert(name.to_owned());
        }
    }

    /// Keeps `pso` and `root` alive until the list is reset and records `name`,
    /// deduplicated per pipeline pointer until reset.
    /// With [`markers`] on: the commands from here to the next pipeline
    /// bound (or the end of the list) in a marker group named `name`, the
    /// title pipeline's shaders.
    pub(crate) fn mark_pipeline(&self, name: &str) {
        if !markers() {
            return;
        }
        self.end_marker();
        let text: Vec<u16> = name.encode_utf16().chain([0]).collect();
        const UNICODE: u32 = 0;
        // SAFETY: an open list; the text is copied when recorded.
        unsafe {
            self.list
                .BeginEvent(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
        };
        self.marked.store(true, Ordering::Relaxed);
    }

    fn end_marker(&self) {
        if self.marked.swap(false, Ordering::Relaxed) {
            // SAFETY: an open list with the group `mark_pipeline` began.
            unsafe { self.list.EndEvent() };
        }
    }

    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound_pipelines.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.defer_title_copy(Step::Copy(dst as usize, src as usize, len));
    }

    /// Moves bytes between overlapping title ranges at submission.
    pub(crate) fn defer_move(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.defer_title_copy(Step::Move(dst as usize, src as usize, len));
    }

    fn defer_title_copy(&self, step: Step) {
        let mut deferred = self.deferred();
        deferred.copies.push(step);
        deferred.refreshed.clear();
        deferred.slots.clear();
        deferred.filled_scratch.clear();
        self.changed_deferred();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Host GPU address of scratch memory holding `len` bytes from `src` as of
    /// submission, reused across draws until a title copy or list reset.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        if let Some(&gpu) = self.deferred().filled_scratch.get(&key) {
            return Some(gpu);
        }
        let (gpu, to) = self.scratch(len)?;
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Fill(to as usize, src as usize, len as usize));
        deferred.filled_scratch.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Brings the buffer range of the view currently in `index` up to date
    /// when the list is submitted, once until a title write intervenes.
    ///
    /// Measuring: one step per slot a table draw reads (the set above keeps
    /// repeats out), timed with the refreshes in `StepRefreshUs`, and a slot
    /// whose buffer is clean adds nothing to the batch.
    pub(crate) fn defer_slot(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>, index: u32) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, index);
        if deferred.slots.insert(key) {
            deferred.copies.push(Step::Slot(viewed.clone(), index));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let memory = self.memory_writer.load(Ordering::Relaxed);
        {
            let mut deferred = self.deferred();
            deferred.has_gpu_writes = true;
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len, marked, memory));
        }
        self.clamps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hazard
            .note_write(address as u64, len as u64);
    }

    /// Notes that the list binds the view write bitmaps: its shaders may
    /// mark what they write ([`Self::marks_writes`]).
    pub(crate) fn note_marks(&self) {
        self.deferred().marks = true;
    }

    /// Notes that an earlier command in this list may write arbitrary title
    /// buffers on the GPU (bindless heap UAVs or page-table memory writes).
    pub(crate) fn note_unknown_gpu_write(&self) {
        self.clamps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hazard
            .note_unknown();
    }

    /// Notes that an inline indirect clamp was recorded in this list so later
    /// clamps stay inline and charge the list budget in recorded order.
    pub(crate) fn note_inline_clamp(&self) {
        self.clamps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hazard
            .note_inline();
    }

    /// Whether an indirect execution reading `[arguments, arguments + len)`
    /// (and optional 4-byte `count`) has no earlier conflicting GPU write or
    /// inline clamp in this list and may be clamped in `prelude`.
    #[cfg(test)]
    pub(crate) fn can_hoist_clamp(&self, arguments: u64, len: u64, count: Option<u64>) -> bool {
        self.clamp_blocker(arguments, len, count).is_none()
    }

    /// What keeps the clamp of an indirect execution reading
    /// `[arguments, arguments + len)` (and the 4-byte `count`) out of
    /// `prelude`: `None` if it may be hoisted there.
    pub(crate) fn clamp_blocker(
        &self,
        arguments: u64,
        len: u64,
        count: Option<u64>,
    ) -> Option<ClampBlock> {
        self.clamps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hazard
            .blocker(arguments, len, count)
    }

    /// The list was started for a clamp the list before it could not
    /// hoist (`crate::list::cut_for_clamp`).
    pub(crate) fn mark_clamp_cut(&self) {
        self.clamps.lock().unwrap_or_else(|e| e.into_inner()).cut = true;
    }

    /// Whether a clamp this list cannot hoist is worth a new list of its
    /// own: always, unless this list was itself started for a clamp and
    /// took fewer than [`CUT_PAYS`] into its prelude before the next write
    /// blocked one. Such a cut bought nothing over clamping inline (one
    /// compute pass and one restarted render pass either way), and a title
    /// that writes before every execution would cut before every one, so
    /// the rest of the list clamps inline as before.
    pub(crate) fn clamp_cut_pays(&self) -> bool {
        let clamps = self.clamps.lock().unwrap_or_else(|e| e.into_inner());
        !clamps.cut || clamps.pending.len() >= CUT_PAYS
    }

    /// Allocates a 16-byte-aligned slice of `bytes` in this list's hoisted
    /// indirect argument arena and records its prelude clamp.
    pub(crate) fn enqueue_hoisted_clamp(
        &self,
        bytes: u64,
        make: impl FnOnce(u64) -> crate::indirect::PendingClamp,
    ) -> Option<(ID3D12Resource, u64)> {
        let budget_gpu = self.budget()?;
        let host = host()?;
        let mut clamps = self.clamps.lock().unwrap_or_else(|e| e.into_inner());
        let (buffer, offset, dst_gpu) =
            clamps.arena.alloc(host, bytes, "hoisted indirect arguments")?;
        clamps.budget_gpu = budget_gpu;
        clamps.pending.push(make(dst_gpu));
        Some((buffer, offset))
    }

    /// Whether scoped Enhanced Barriers (`VICTORY_CLAMP_SYNC=1`) are active
    /// for indirect clamps on this list.
    pub(crate) fn narrow_clamp_sync(&self) -> bool {
        crate::indirect::clamp_sync_enabled() && self.list7.is_some() && self.prelude7.is_some()
    }

    /// Cached `ID3D12GraphicsCommandList7` for `self.list` when narrow clamp
    /// sync is active.
    pub(crate) fn narrow_clamp_sync_list(&self) -> Option<&ID3D12GraphicsCommandList7> {
        self.narrow_clamp_sync()
            .then_some(self.list7.as_ref())
            .flatten()
    }

    /// Cached `ID3D12GraphicsCommandList7` for `self.prelude.1` when narrow
    /// clamp sync is active.
    pub(crate) fn narrow_clamp_sync_prelude(&self) -> Option<&ID3D12GraphicsCommandList7> {
        self.narrow_clamp_sync()
            .then_some(self.prelude7.as_ref())
            .flatten()
    }

    /// Bump-allocates a disjoint 16-byte-aligned inline argument slice from
    /// this list's inline scratch arena (zero WAR hazard across consecutive
    /// inline clamps in narrow sync mode).
    pub(crate) fn inline_argument_slice(&self, bytes: u64) -> Option<(ID3D12Resource, u64, u64)> {
        let host = host()?;
        let mut arena = self
            .inline_arguments
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        arena.alloc(host, bytes, "inline indirect arguments")
    }

    /// Transitions hoisted indirect argument buffers back to `COMMON` at the
    /// end of `list.list` so the list can be re-submitted or reset cleanly.
    /// No-op when narrow clamp sync is active (buffers stay in `COMMON`).
    fn hoisted_clamps_close(&self) {
        if self.narrow_clamp_sync() {
            return;
        }
        let buffers = {
            let clamps = self.clamps.lock().unwrap_or_else(|e| e.into_inner());
            if clamps.pending.is_empty() {
                return;
            }
            clamps.touched_buffers()
        };
        if buffers.is_empty() {
            return;
        }
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = buffers
            .iter()
            .map(|b| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        // SAFETY: an open list and live argument buffers left in
        // `INDIRECT_ARGUMENT` by `prelude`.
        unsafe {
            self.list.ResourceBarrier(&barriers);
        }
    }

    /// Resets hoisted clamp tracking after the list finishes running, keeping
    /// the chunks needed by the last cycle (plus one) for zero-allocation reuse.
    fn reset_hoisted_clamps(&self) {
        let (spent_chunks, spent_oversized) = {
            let mut clamps = self.clamps.lock().unwrap_or_else(|e| e.into_inner());
            clamps.pending.clear();
            clamps.budget_gpu = 0;
            clamps.hazard.clear();
            clamps.cut = false;
            clamps.arena.reset()
        };
        if !spent_chunks.is_empty() {
            release_later(Box::new(spent_chunks));
        }
        if !spent_oversized.is_empty() {
            release_later(Box::new(spent_oversized));
        }
    }

    /// Resets the inline indirect argument scratch arena after the list
    /// finishes running, keeping recently used chunks for zero-allocation reuse.
    fn reset_inline_arguments(&self) {
        let (spent_chunks, spent_oversized) = self
            .inline_arguments
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reset();
        if !spent_chunks.is_empty() {
            release_later(Box::new(spent_chunks));
        }
        if !spent_oversized.is_empty() {
            release_later(Box::new(spent_oversized));
        }
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.has_heaps = true;
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(Box::new(work));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory value the list waits for before it starts.
    pub(crate) fn defer_wait(&self, wait: crate::hold::ValueWait) {
        self.deferred().waits.push(wait);
    }

    /// The memory values the list waits for before it starts.
    pub(crate) fn waits(&self) -> Vec<crate::hold::ValueWait> {
        self.deferred().waits.clone()
    }

    /// Whether a deferred bottom-of-pipe value write recorded on this list
    /// satisfies `wait` once the list finishes.
    pub(crate) fn will_write(&self, wait: &crate::hold::ValueWait) -> bool {
        let addr = wait.address as usize;
        let need = usize::from(wait.bytes);
        self.deferred()
            .writes
            .iter()
            .rev()
            .find(|&&(at, _, n)| at == addr && n >= need)
            .is_some_and(|&(_, what, _)| match what {
                crate::queue::Written::Value(v) => wait.holds_for(v) == Some(true),
                crate::queue::Written::GpuTimestamp | crate::queue::Written::SystemTimestamp => {
                    false
                }
            })
    }

    /// Summary of deferred CPU/GPU side effects on this list for submission
    /// coalescing (`crate::hold::can_join_batch`).
    /// Whether the list's landing changes nothing a later submission's
    /// preparation reads: no ranges or texels go back to title memory, no
    /// shader marks, no work after it or bottom-of-pipe writes, and no
    /// staging of its own uploads still read (`VICTORY_PREP_AHEAD`).
    pub(crate) fn quiet_at_landing(&self) -> bool {
        let marks = self.marks_writes();
        let d = self.deferred();
        !marks
            && d.after.is_empty()
            && d.writes.is_empty()
            && d.texture_backs.is_empty()
            && d.back_wanted.is_empty()
            && !d
                .copies
                .iter()
                .any(|s| matches!(s, Step::GpuWrite(..) | Step::Texture(..)))
    }

    /// Whether the list's shaders may mark what they write in the view
    /// write bitmaps, which every landing takes
    /// ([`crate::shadow::copy_back_marks`]).
    /// Why the list is not [`Self::quiet_at_landing`], as [`flight::NotQuiet`]
    /// bits (the `VICTORY_PREP_AHEAD` probe).
    pub(crate) fn not_quiet_reasons(&self) -> u8 {
        use flight::NotQuiet as N;
        let marks = self.marks_writes();
        let d = self.deferred();
        let mut bits = 0;
        let mut set = |r: N, on: bool| bits |= u8::from(on) << r as u8;
        set(N::Declared, d.copies.iter().any(|s| matches!(s, Step::GpuWrite(_, _, _, false, _))));
        set(N::Marks, marks);
        set(N::After, !d.after.is_empty());
        set(N::BottomOfPipe, !d.writes.is_empty());
        set(N::ReadBack, !d.texture_backs.is_empty() || !d.back_wanted.is_empty());
        set(N::Textures, d.copies.iter().any(|s| matches!(s, Step::Texture(..))));
        bits
    }

    /// Adds what the list's landing writes to `w`
    /// (`VICTORY_PREP_AHEAD=ranges`); `false` if that is not known before it
    /// runs: work after it, read-backs, table textures, or a bottom-of-pipe
    /// write into a host copy (shaders after it would read the copy's older
    /// bytes until a refresh).
    pub(crate) fn ahead_writes(&self, w: &mut flight::Writes) -> bool {
        use flight::NotQuiet as N;
        let known = (1 << N::Declared as u8) | (1 << N::Marks as u8) | (1 << N::BottomOfPipe as u8);
        if self.not_quiet_reasons() & !known != 0 {
            return false;
        }
        if self.marks_writes() {
            w.marks();
        }
        let d = self.deferred();
        for &(at, _, bytes) in &d.writes {
            if crate::shadow::find(at as u64, bytes as u64).is_some() {
                return false;
            }
            w.span(at, at.saturating_add(bytes));
        }
        for step in &d.copies {
            if let Step::GpuWrite(shadow, address, len, marked, _) = step {
                let (copy, (start, size)) = (std::sync::Arc::as_ptr(shadow) as usize, shadow.range());
                if *marked {
                    w.whole(copy, start as usize, (start + size) as usize);
                } else {
                    w.declared(copy, *address, address.saturating_add(*len));
                }
            }
        }
        true
    }

    pub(crate) fn marks_writes(&self) -> bool {
        let unknown = self.clamps.lock().unwrap_or_else(|e| e.into_inner()).hazard.unknown;
        let d = self.deferred();
        unknown
            || d.marks
            || !d.missed.is_empty()
            || d.copies.iter().any(|s| matches!(s, Step::GpuWrite(_, _, _, true, _)))
    }

    pub(crate) fn batch_flags(&self) -> crate::hold::BatchFlags {
        let d = self.deferred();
        let mut flags = crate::hold::BatchFlags {
            has_after: !d.after.is_empty(),
            ..Default::default()
        };
        for step in &d.copies {
            match step {
                Step::Copy(..) | Step::Move(..) => {
                    flags.has_title_copy = true;
                    flags.has_cpu_read = true;
                }
                Step::Fill(..) => flags.has_cpu_read = true,
                Step::GpuWrite(..) => flags.has_gpu_write = true,
                Step::Texture(..) => flags.has_textures = true,
                Step::Refresh(..)
                | Step::Slot(..)
                | Step::Host(_)
                | Step::Heap(_)
                | Step::Memory(_) => {}
            }
        }
        flags
    }

    /// Whether nothing has been recorded since the list was made or reset
    /// that a wait recorded now would have to follow: no work, texture
    /// use, bottom-of-pipe write or pipeline (state settings alone carry
    /// over).
    pub(crate) fn is_empty(&self) -> bool {
        let d = self.deferred();
        let deferred_empty = d.copies.is_empty() && d.uploads.is_empty() && d.writes.is_empty() && d.after.is_empty() && d.missed.is_empty();
        drop(d);
        deferred_empty
            && self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.clamps.lock().unwrap_or_else(|e| e.into_inner()).pending.is_empty()
    }

    /// `size` bytes of mapped upload memory for a texture to copy from,
    /// alive until the list is reset: the buffer, the offset in it (a
    /// multiple of 512, as texture copies need) and the CPU pointer. It is
    /// the list's scratch, not a buffer made for each copy (fresh upload
    /// buffers cost ~5 ms per 12.7 MB, run textures8 in `crate::texture`),
    /// except for a copy larger than a chunk.
    pub(crate) fn upload(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let counts = &crate::list::FOOTPRINT_UPLOADS;
            counts[0].fetch_add(1, Relaxed);
            counts[1].fetch_add(size, Relaxed);
            let own = size.max(1).next_multiple_of(256) > SCRATCH_CHUNK;
            counts[2].fetch_add(u64::from(own), Relaxed);
        }
        let (buffer, at, _, data) = self.scratch_in(size, 512)?;
        Some((buffer, at, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.readback_buffer(size)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (_, _, gpu, data) = self.scratch_in(len, 256)?;
        Some((gpu, data))
    }

    /// `len` bytes of upload memory at a multiple of `align` (a power of
    /// two) until the list is reset: the buffer, the offset in it, the host
    /// GPU address and the CPU pointer.
    fn scratch_in(&self, len: u64, align: u64) -> Option<(ID3D12Resource, u64, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer.clone());
            return Some((buffer, 0, gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used.next_multiple_of(align).min(chunk.capacity) {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used.next_multiple_of(align);
        let data = (chunk.mapped + at as usize) as *mut u8;
        let result = (chunk.buffer.clone(), at, chunk.gpu + at, data);
        s.used = at + len;
        Some(result)
    }

    /// Starts the next cycle of scratch at the first chunk and gives up
    /// the chunks past the one after the last cycle's last (only after the
    /// list's last submission finished, when the GPU no longer reads them).
    /// A list so keeps what its last cycle needed, and a cycle a little
    /// larger does not allocate again, but a rare peak is not kept for good.
    fn trim_scratch(&self) {
        let spent = {
            let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            let kept = (s.current + 2).min(s.chunks.len());
            let spent = s.chunks.split_off(kept);
            (s.current, s.used) = (0, 0);
            spent
        };
        // Each release is a kernel call, deferred across `bury_lag` submissions.
        if !spent.is_empty() {
            retire_any(spent);
        }
    }

    /// The upload memory the list keeps for its next cycle, in bytes.
    pub(crate) fn scratch_bytes(&self) -> u64 {
        let s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        s.chunks.iter().map(|c| c.capacity).sum()
    }

    /// Gives up all the scratch of a list that is reset and that no title
    /// list or submission uses (`list::SPARE`): the next cycle makes what it
    /// needs again. Released as [`Self::trim_scratch`] releases its chunks.
    pub(crate) fn shed_scratch(&self) {
        let spent = {
            let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (s.current, s.used) = (0, 0);
            std::mem::take(&mut s.chunks)
        };
        // Each release is a kernel call, as in [`release_later`].
        if !spent.is_empty() {
            release_later(Box::new(spent));
        }
    }

    /// The host address of the list's indirect-work budget (plus 64 scratch
    /// dwords at bytes 64..320 for the 64-lane clamp), zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(512)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                (at as *mut u32).write_volatile(0)
            })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// The indirect-work budget word consumed by the list's last execution.
    #[cfg(test)]
    pub(crate) fn budget_used(&self) -> Option<u32> {
        let (_, data) = self.budget.get()?.as_ref()?;
        // SAFETY: the list's own mapped budget word, read after execution finishes.
        Some(unsafe { (*data as *const u32).read_volatile() })
    }

    /// The list's argument buffer, at least `bytes` long, common between
    /// its uses (`crate::indirect`).
    pub(crate) fn arguments(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live buffer.
        if let Some(b) = arguments.as_ref().filter(|b| unsafe { b.GetDesc() }.Width >= bytes) {
            return Some(b.clone());
        }
        let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
        name(&made, || "indirect arguments".into());
        // Commands recorded earlier may still name the smaller one.
        if let Some(old) = arguments.replace(made.clone()) {
            self.keep(&old);
        }
        Some(made)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        let raw = object.as_raw() as usize;
        let mut deferred = self.deferred();
        if !deferred.kept_ptrs.insert(raw) {
            return;
        }
        match object.cast::<windows::core::IUnknown>() {
            Ok(object) => deferred.kept.push(object),
            Err(_) => {
                deferred.kept_ptrs.remove(&raw);
            }
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    pub(crate) fn alias_source_changed(
        &self,
        view: &ID3D12Resource,
        source: &ID3D12Resource,
    ) -> bool {
        let view = view.as_raw() as usize;
        let source = source.as_raw() as usize;
        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// Runs `work` on host memory at submission, in order with the copies.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer and host texture `copy` up to date
    /// when the list is submitted (`crate::bindless::uploads`), keeping
    /// both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        let mut deferred = self.deferred();
        deferred.has_textures = true;
        deferred.copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            // The preparing report counts the copies' time only (the other
            // steps `continue` before it): they read no clock without
            // diagnostics.
            let copy_started = matches!(step, Step::Copy(..) | Step::Move(..) | Step::Fill(..))
                .then(std::time::Instant::now);
            // The list-steps line: every step counted, all but the many
            // refreshes and slots timed.
            let step_kind = match step {
                Step::Copy(..) | Step::Move(..) | Step::Fill(..) => 0,
                Step::Refresh(..) => 1,
                Step::Slot(..) => 2,
                Step::Host(_) => 3,
                Step::GpuWrite(..) => 4,
                Step::Texture(..) => 5,
                Step::Heap(_) => 6,
                Step::Memory(_) => 7,
            };
            let _step = StepTimer(step_kind, (step_kind > 2).then(std::time::Instant::now));
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) | Step::Slot(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Move(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                // `VICTORY_SAME_COPIES=skip`: a copy that changes nothing
                // anywhere is left out (`same_copy`).
                &Step::Copy(dst, src, len)
                    if same_copy::mode() != same_copy::Mode::Off
                        && same_copy::in_place(dst, src, len)
                        && same_copy::mode() == same_copy::Mode::Skip =>
                {
                    0
                }
                &Step::Copy(dst, src, len) | &Step::Move(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_copied(dst, len);
                    if matches!(step, Step::Move(..)) {
                        move_mapped(dst as *mut u8, src as *const u8, len);
                    } else {
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    }
                    if crate::shadow::shadow_keep_enabled() {
                        crate::shadow::forget_range(dst, len);
                        for page in crate::watch::alias_pages(dst, len) {
                            crate::shadow::forget_range(
                                page * crate::watch::PAGE,
                                crate::watch::PAGE,
                            );
                        }
                    }
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    flight::before_cpu_read(Some((src, len)));
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Slot(viewed, index) => {
                    crate::bindless::refresh_slot(viewed, *index, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                Step::Host(work) => {
                    flight::before_cpu_read(None);
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    flight::before_gpu_write(
                        std::sync::Arc::as_ptr(shadow) as usize,
                        *address,
                        *len,
                    );
                    if !marked {
                        batch.flush();
                    }
                    shadow.writing(*address, *len, *marked);
                    continue;
                }
                // Brought up to date before the submission's lists were made.
                Step::Texture(..) => continue,
                Step::Heap(viewed) => {
                    if batch.need_heap(viewed) {
                        crate::bindless::refresh(viewed, |s, a, l| batch.add(s, a, l));
                    }
                    continue;
                }
                &Step::Memory(heap) => {
                    if batch.need_memory(heap) {
                        crate::memory::prepare(heap, |s, a, l| batch.add(s, a, l));
                    }
                    continue;
                }
            };
            if let Some(copy_started) = copy_started {
                let us = copy_started.elapsed().as_micros() as u64;
                PREPARING[0].fetch_add(us, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                let kind = if matches!(step, Step::Fill(..)) { 3 } else { 0 };
                COPY_KINDS[kind].fetch_add(1, Ordering::Relaxed);
                COPY_KINDS[kind + 1].fetch_add(us, Ordering::Relaxed);
                COPY_KINDS[kind + 2].fetch_add(copied as u64, Ordering::Relaxed);
            }
        }
    }
}

/// A submission's refreshes not done yet, as the ranges of each buffer copy
/// asked for (ranges less than [`JOIN`] apart are done as one): done before
/// a step that writes title memory or gives a range to shaders, and before
/// the lists run.
///
/// A copy's ranges brought up to date are not copied again in the
/// submission until a step writes them: every list of a world submission
/// using title memory asks for every copy whole, and the title copies into
/// buffers ~80 times in between. A step that may write anywhere
/// ([`Batch::title_written`]) makes all of them due again, a title copy
/// ([`Batch::title_copied`]) only what it writes. What the title's threads
/// write meanwhile is for later submissions (on the console it would race
/// the GPU).
#[derive(Default)]
struct PtrHasher(u64);

impl std::hash::Hasher for PtrHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(b);
        }
    }
    fn write_usize(&mut self, i: usize) {
        let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = x ^ (x >> 32);
    }
}

type PtrMap<V> = std::collections::HashMap<usize, V, std::hash::BuildHasherDefault<PtrHasher>>;

struct BatchBuffers {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    pending: PtrMap<usize>,
    current: PtrMap<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    extents: victory_formats::extents::Extents,
    pending_extents: victory_formats::extents::Extents,
    heaps_seen: Vec<(usize, u64)>,
    memories_seen: Vec<(usize, u64)>,
}

static BATCH_POOL: Mutex<Vec<BatchBuffers>> = Mutex::new(Vec::new());

#[derive(Default)]
struct Batch {
    /// The ranges still to bring up to date, by copy.
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    /// Where each copy's refreshes are in `refreshes`, by copy.
    pending: PtrMap<usize>,
    /// The ranges of each copy brought up to date since the last write that
    /// can have changed them, by copy (held, so that its address names no
    /// other).
    current: PtrMap<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    /// Totals and the largest ranges refreshed by this submission, for a
    /// bounded diagnostics line. No range contents or view attribution.
    #[cfg(feature = "diagnostics")]
    summary: crate::ranges::RefreshSummary,
    /// Where each copy in `current` sits in memory, by copy: a title copy
    /// reaches by its bytes only the copies that can hold them, not
    /// every copy flushed.
    extents: victory_formats::extents::Extents,
    /// As `extents`, for the copies whose refreshes are still pending: a
    /// title copy flushes only the ones its bytes can be in.
    pending_extents: victory_formats::extents::Extents,
    /// Bindless heaps (`(viewed_ptr, hull_generation)`) already refreshed
    /// since the last title write or copy.
    heaps_seen: Vec<(usize, u64)>,
    /// Title-memory page tables (`(heap_key, shadow_generation)`) already
    /// prepared since the last title write or copy.
    memories_seen: Vec<(usize, u64)>,
}

/// How far apart a copy's pending ranges may be and still be done as one,
/// the bytes between copied too: a call that copies a range costs about
/// what copying this much does at the 6-7 GB/s testers saw (a guess, not
/// measured).
const JOIN: usize = crate::watch::PAGE;

impl Drop for Batch {
    fn drop(&mut self) {
        self.refreshes.clear();
        self.pending.clear();
        self.current.clear();
        self.extents.clear();
        self.pending_extents.clear();
        self.heaps_seen.clear();
        self.memories_seen.clear();
        if self.current.capacity() == 0
            && self.pending.capacity() == 0
            && self.heaps_seen.capacity() == 0
            && self.memories_seen.capacity() == 0
        {
            return;
        }
        let mut pool = BATCH_POOL.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < 4 {
            pool.push(BatchBuffers {
                refreshes: std::mem::take(&mut self.refreshes),
                pending: std::mem::take(&mut self.pending),
                current: std::mem::take(&mut self.current),
                extents: std::mem::take(&mut self.extents),
                pending_extents: std::mem::take(&mut self.pending_extents),
                heaps_seen: std::mem::take(&mut self.heaps_seen),
                memories_seen: std::mem::take(&mut self.memories_seen),
            });
        }
    }
}

impl Batch {
    fn take() -> Self {
        let Some(buffers) = BATCH_POOL.lock().unwrap_or_else(|e| e.into_inner()).pop() else {
            return Self::default();
        };
        Self {
            refreshes: buffers.refreshes,
            pending: buffers.pending,
            current: buffers.current,
            #[cfg(feature = "diagnostics")]
            summary: Default::default(),
            extents: buffers.extents,
            pending_extents: buffers.pending_extents,
            heaps_seen: buffers.heaps_seen,
            memories_seen: buffers.memories_seen,
        }
    }

    fn need_heap(&mut self, viewed: &crate::bindless::Viewed) -> bool {
        let key = (std::ptr::from_ref(viewed) as usize, viewed.hull_generation());
        if self.heaps_seen.contains(&key) {
            return false;
        }
        if let Some(entry) = self.heaps_seen.iter_mut().find(|(p, _)| *p == key.0) {
            entry.1 = key.1;
        } else {
            self.heaps_seen.push(key);
        }
        true
    }

    fn need_memory(&mut self, heap: usize) -> bool {
        let key = (heap, crate::shadow::generation());
        if self.memories_seen.contains(&key) {
            return false;
        }
        if let Some(entry) = self.memories_seen.iter_mut().find(|(h, _)| *h == heap) {
            entry.1 = key.1;
        } else {
            self.memories_seen.push(key);
        }
        true
    }
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        // Nothing to copy: a world submission asks for every copy its heaps'
        // views and page tables reach (~9,000), a few of them written.
        if shadow.clean() {
            return;
        }
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        flight::probe_refresh(key, address, len);
        flight::before_refresh(key, address, len);
        // Only what is not current: most lists ask for a copy whole, after
        // a title copy took a few bytes of it.
        let empty = crate::ranges::Ranges::default();
        let have = self.current.get(&key).map_or(&empty, |(_, have)| have);
        let mut todo = have.missing(address, end).peekable();
        if todo.peek().is_none() {
            // Already current: what a later list asked again, and what
            // #259 keeps current across its moves.
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, (end - address) as u64);
            }
            return;
        }
        let i = match self.pending.get(&key) {
            Some(&i) => i,
            None => {
                let (start, len) = shadow.range();
                self.pending_extents
                    .flushed(key, start as usize, (start + len) as usize);
                self.refreshes.push((shadow.clone(), Default::default()));
                let i = self.refreshes.len() - 1;
                self.pending.insert(key, i);
                i
            }
        };
        for (from, to) in todo {
            self.refreshes[i].1.insert(from, to, JOIN);
        }
    }

    /// Before a step that may write any title memory: pending refreshes are
    /// done first, and copies are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.current.clear();
        self.extents.clear();
        self.heaps_seen.clear();
        self.memories_seen.clear();
    }

    /// As [`Batch::title_written`], before a copy into the title memory
    /// `[at, at + len)`, which `copy_mapped` writes and nothing more: only
    /// what the copy can have changed is checked again after it.
    // shortcut: memory shown at two addresses is what the runtime said it
    // maps twice (`crate::watch::aliased`, as the trackers know it); where
    // it says so, the copy makes everything due again, as it may change
    // bytes any copy holds.
    fn title_copied(&mut self, at: usize, len: usize) {
        if crate::watch::aliased(at, len) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::AliasedCopies, 1);
            return self.title_written();
        }
        let (from, to) = touched(at, at.saturating_add(len));
        if from >= to {
            return;
        }
        // A non-aliased copy still marks its destination shadow dirty, so
        // subsequent `Step::Heap` and `Step::Memory` steps must re-scan to
        // pick up the post-copy bytes.
        self.heaps_seen.clear();
        self.memories_seen.clear();
        // Flush only pending refreshes whose ranges overlap the destination
        // window; disjoint copies stay queued so they coalesce across lists.
        // The index reaches the copies whose memory can hold the window's
        // bytes; the rest are left queued without a look.
        let mut overlapping = Vec::new();
        let mut stale = 0;
        for key in self.pending_extents.overlapping(from, to) {
            let Some(&i) = self.pending.get(&key) else {
                stale += 1;
                continue;
            };
            let (_, ranges) = &self.refreshes[i];
            if !ranges.iter().any(|(r_from, r_to)| r_from < to && r_to > from) {
                continue;
            }
            let (shadow, ranges) = self.refreshes.swap_remove(i);
            self.pending.remove(&key);
            if let Some((moved, _)) = self.refreshes.get(i) {
                self.pending
                    .insert(std::sync::Arc::as_ptr(moved) as usize, i);
            }
            overlapping.push((shadow, ranges));
        }
        // The entries of the copies flushed above name nothing pending any
        // more; once they outnumber the live ones, filing the live ones
        // again costs less than walking past the dead ones.
        if stale > self.refreshes.len() {
            self.pending_extents.clear();
            for (shadow, _) in &self.refreshes {
                let (start, len) = shadow.range();
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                self.pending_extents
                    .flushed(key, start as usize, (start + len) as usize);
            }
        }
        if !overlapping.is_empty() {
            self.flush_work(overlapping);
        }
        // Only the copies whose memory the bytes can be in: the rest
        // hold nothing of the window, and what a no-op remove would
        // leave whole it leaves whole.
        for key in self.extents.overlapping(from, to) {
            if let Some((_, have)) = self.current.get_mut(&key) {
                have.remove(from, to);
            }
        }
    }

    /// Does the refreshes; returns the bytes copied.
    fn flush(&mut self) -> usize {
        self.pending.clear();
        self.pending_extents.clear();
        let work = std::mem::take(&mut self.refreshes);
        self.flush_work(work)
    }

    /// Runs `work` refreshes, recording their covered ranges into `current`
    /// and `extents`; returns the bytes copied.
    fn flush_work(
        &mut self,
        work: Vec<(std::sync::Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    ) -> usize {
        if work.is_empty() {
            return 0;
        }
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        let total_requested: usize = work
            .iter()
            .flat_map(|(_, ranges)| ranges.iter())
            .map(|(from, to)| to - from)
            .sum();
        let mut total = 0;
        #[cfg(feature = "diagnostics")]
        let mut counted = crate::copystat::hold();
        if crate::shadow::parallel_copies_enabled()
            && work.len() >= 2
            && total_requested >= crate::shadow::PARALLEL_REFRESH_MIN_BYTES
        {
            type RefreshOut = Vec<(usize, usize, usize, u64)>;
            let results: Vec<Mutex<RefreshOut>> =
                (0..work.len()).map(|_| Mutex::default()).collect();
            crate::texture::in_parallel(work.len(), |i| {
                let (shadow, ranges) = &work[i];
                let collected = ranges.as_slice();
                let started = std::time::Instant::now();
                let copied = shadow.update_ranges(collected);
                // The ranges are copied as one batch, so they are timed as
                // one: its time is the first range's, the others none.
                let us = started.elapsed().as_micros() as u64;
                let mut out = Vec::with_capacity(collected.len());
                for (n, (&(from, to), copied)) in collected.iter().zip(copied).enumerate() {
                    out.push((from, to, copied, if n == 0 { us } else { 0 }));
                }
                *results[i].lock().unwrap_or_else(|e| e.into_inner()) = out;
            });
            for ((shadow, _), done) in work.into_iter().zip(results) {
                let key = std::sync::Arc::as_ptr(&shadow) as usize;
                let (start, len) = shadow.range();
                let done = done.into_inner().unwrap_or_else(|e| e.into_inner());
                crate::shadow::note_refreshed(
                    &shadow,
                    done.iter().map(|d| d.2).sum(),
                    done.iter().map(|d| d.3).sum(),
                );
                for (from, to, copied, us) in done {
                    #[cfg(feature = "diagnostics")]
                    {
                        counted.range(from, to - from, us, 0);
                        self.summary
                            .record(from as u64, (to - from) as u64, shadow.gpu(from as u64));
                    }
                    PREPARING[0].fetch_add(us, Ordering::Relaxed);
                    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                    total += copied;
                    let (from, to) = taken(from, to, (start as usize, (start + len) as usize));
                    let (_, have) = match self.current.entry(key) {
                        std::collections::hash_map::Entry::Vacant(vacant) => {
                            self.extents
                                .flushed(key, start as usize, (start + len) as usize);
                            vacant.insert((shadow.clone(), Default::default()))
                        }
                        std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                    };
                    have.insert(from, to, 0);
                }
            }
            return total;
        }
        for (shadow, ranges) in work {
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            let (start, len) = shadow.range();
            let collected = ranges.as_slice();
            let started = std::time::Instant::now();
            let copied = shadow.update_ranges(collected);
            // As above: the batch is timed as one, on its first range.
            let us = started.elapsed().as_micros() as u64;
            crate::shadow::note_refreshed(&shadow, copied.iter().sum(), us);
            for (n, (&(from, to), copied)) in collected.iter().zip(copied).enumerate() {
                let us = if n == 0 { us } else { 0 };
                #[cfg(feature = "diagnostics")]
                {
                    counted.range(from, to - from, us, 0);
                    self.summary
                        .record(from as u64, (to - from) as u64, shadow.gpu(from as u64));
                }
                PREPARING[0].fetch_add(us, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                total += copied;
                let (from, to) = taken(from, to, (start as usize, (start + len) as usize));
                // The copy's memory joins the index as its first range
                // joins `current`: a later title copy reaches it by its
                // bytes.
                let (_, have) = match self.current.entry(key) {
                    std::collections::hash_map::Entry::Vacant(vacant) => {
                        self.extents
                            .flushed(key, start as usize, (start + len) as usize);
                        vacant.insert((shadow.clone(), Default::default()))
                    }
                    std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                };
                have.insert(from, to, 0);
            }
        }
        total
    }

    /// Logs one compact line for the submission, if it refreshed title memory.
    #[cfg(feature = "diagnostics")]
    fn log_refreshes(&self, submission: u64, lists: usize) {
        if !self.summary.is_empty() {
            tracing::info!(
                submission,
                lists,
                "title-memory refreshes: {}",
                self.summary
            );
        }
    }
}

/// What a write to title memory `[from, to)` changes in the trackers: the
/// 64 KiB pages it touches, whole (`crate::watch::writing`); just the bytes
/// where nothing is tracked.
fn touched(from: usize, to: usize) -> (usize, usize) {
    let page = crate::watch::PAGE;
    if crate::watch::fault_tracking() {
        (from / page * page, to.div_ceil(page).saturating_mul(page))
    } else {
        (from, to)
    }
}

/// What a refresh of `[from, to)` of the copy of title memory `memory`
/// (where it starts and ends) leaves current: all of it, but with tracking
/// only the pages it took, those it holds whole (the ends of the copy's
/// memory count whole, as `Tracked::take` counts them). A page it holds in
/// part stays changed, and the next request that holds it whole must take
/// it, or copy-back reads it as written by the title and keeps the
/// shaders' writes there out of title memory.
fn taken(from: usize, to: usize, memory: (usize, usize)) -> (usize, usize) {
    let page = crate::watch::PAGE;
    if crate::watch::fault_tracking() {
        (
            if from <= memory.0 {
                from
            } else {
                from.next_multiple_of(page)
            },
            if to >= memory.1 { to } else { to / page * page },
        )
    } else {
        (from, to)
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::MoveUs,
        moved.elapsed().as_micros() as u64,
    );
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(test)]
    COPY_MAPPED_FALLBACKS.with(|n| n.set(n.get() + 1));
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FallbackUs);
    // The whole copy failed. Copy each readable region in one call and
    // zero the holes. In particular, a hole that starts inside a 64 KiB
    // block does not require a faulting read of each remaining 4 KiB page.
    let (start, end) = (src as usize, src as usize + len);
    let mut at = start;
    while at < end {
        let (region_end, readable) = source_region(at);
        let next = region_end.min(end);
        let to = dst.wrapping_add(at - start);
        if !readable {
            // Wine's view of memory can omit host mappings. Keep trying
            // the initial 64 KiB piece before treating the region as a hole.
            let piece = ((at >> 16) + 1).saturating_mul(1 << 16).min(end);
            if read(at as *const u8, to, piece - at) {
                at = piece;
                continue;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (next - at) as u64);
            // SAFETY: inside the destination, which is mapped.
            unsafe { to.write_bytes(0, next - at) };
        } else if !read(at as *const u8, to, next - at) {
            // Protection can change after the query. Keep trying smaller
            // pieces and zero only the pages the kernel cannot read.
            for_pieces(at, next - at, 64 << 10, |block, len| {
                let to = dst.wrapping_add(block - start);
                if !read(block as *const u8, to, len) {
                    for_pieces(block, len, 4 << 10, |page, len| {
                        let to = dst.wrapping_add(page - start);
                        if !read(page as *const u8, to, len) {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Unmapped, len as u64);
                            // SAFETY: inside the mapped destination.
                            unsafe { to.write_bytes(0, len) };
                        }
                    });
                }
            });
        }
        at = next;
    }
}

#[cfg(test)]
thread_local! {
    static COPY_MAPPED_FALLBACKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Number of `copy_mapped` calls on the current thread that fell back to
/// piece-by-piece region reads after encountering an unreadable page.
#[cfg(test)]
pub(crate) fn copy_mapped_fallbacks() -> u64 {
    COPY_MAPPED_FALLBACKS.with(std::cell::Cell::get)
}

/// Moves bytes between overlapping title ranges when both are committed.
fn move_mapped(dst: *mut u8, src: *const u8, len: usize) {
    if crate::texture::committed(src, len) && crate::texture::committed(dst, len) {
        let _writing = crate::watch::writing(dst as usize, len);
        // SAFETY: both ranges name committed title memory. The copy
        // preserves the source bytes when the ranges overlap.
        unsafe { std::ptr::copy(src, dst, len) };
    }
}

/// The end and readability of the region containing `at`.
fn source_region(at: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let page_end = (at | 0xfff).saturating_add(1);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return (page_end, false);
    }
    let readable = info.State == MEM_COMMIT
        && info.Protect != 0
        && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
    let end = (info.BaseAddress as usize)
        .saturating_add(info.RegionSize)
        .max(page_end);
    (end, readable)
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Whether any page in `[addr, addr + len)` is mapped and readable.
pub(crate) fn has_mapped_memory(addr: u64, len: u64) -> bool {
    if addr == 0 || len == 0 {
        return false;
    }
    let mut probe = [0u8; 4];
    if read_into(addr as *const u8, &mut probe[..len.min(4) as usize]) {
        return true;
    }
    let start = addr as usize;
    let end = start.saturating_add(usize::try_from(len).unwrap_or(usize::MAX));
    let mut at = start;
    while at < end {
        let (region_end, readable) = source_region(at);
        if readable && read_into(at as *const u8, &mut probe[..1]) {
            return true;
        }
        at = region_end.min(end);
    }
    // Wine's VirtualQuery can omit host mappings: probe the tail too
    // (streamed textures often populate lower mips at the end first).
    if len > 4 {
        let tail = end.saturating_sub(4);
        if read_into(tail as *const u8, &mut probe) {
            return true;
        }
    }
    false
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn long_unmapped_stretches_read_as_zeros() {
        // 64 reserved blocks: the first and 8 KiB of the second committed,
        // and the last: a hole starting inside a block and spanning many.
        const N: usize = 64;
        let commit = |at: *mut u8, len: usize, fill: u8| {
            // SAFETY: inside the reservation below, ours to fill.
            unsafe {
                assert!(!VirtualAlloc(at.cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, len);
            }
        };
        // SAFETY: a fresh reservation, released below.
        let base = unsafe { VirtualAlloc(std::ptr::null(), N * BLOCK, MEM_RESERVE, PAGE_NOACCESS) };
        let base = base.cast::<u8>();
        assert!(!base.is_null());
        commit(base, BLOCK + (8 << 10), 7);
        // SAFETY: the last block of the reservation.
        commit(unsafe { base.add((N - 1) * BLOCK) }, BLOCK, 9);
        // Neither end aligned.
        let mut to = vec![0xeeu8; N * BLOCK - 200];
        // SAFETY: inside the reservation.
        copy_mapped(
            to.as_mut_ptr(),
            unsafe { base.add(100) },
            N * BLOCK - 200,
            false,
        );
        let (hole, last) = (BLOCK + (8 << 10) - 100, (N - 1) * BLOCK - 100);
        assert!(to[..hole].iter().all(|&b| b == 7));
        assert!(to[hole..last].iter().all(|&b| b == 0));
        assert!(to[last..].iter().all(|&b| b == 9));
        release(base);
    }

    #[test]
    fn protected_holes_inside_a_copy_preserve_the_readable_bytes() {
        const PAGE: usize = 4 << 10;
        // SAFETY: a fresh allocation, filled and protected by this test.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
                .cast::<u8>()
        };
        assert!(!base.is_null());
        let mut old = 0;
        // SAFETY: these are pages of the allocation above.
        unsafe {
            base.write_bytes(7, 4 * BLOCK);
            for offset in [3 * PAGE, BLOCK + PAGE] {
                assert_ne!(
                    VirtualProtect(base.add(offset).cast(), PAGE, PAGE_NOACCESS, &mut old),
                    0
                );
            }
        }
        let mut to = vec![0xee; 4 * BLOCK - 200];
        copy_mapped(to.as_mut_ptr(), base.wrapping_add(100), to.len(), false);
        for (i, byte) in to.into_iter().enumerate() {
            let page = (i + 100) / PAGE;
            let expected = if [3, BLOCK / PAGE + 1].contains(&page) { 0 } else { 7 };
            assert_eq!(byte, expected, "byte {i:#x}");
        }
        release(base);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl HostMemory for Host {
    type Buffer = ID3D12Resource;

    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    /// A buffer in the upload heap, mapped for writing.
    fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A buffer in the readback heap, mapped for reading, in the
    /// copy-destination state.
    fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, true)
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        #[cfg(feature = "diagnostics")]
        let heap = if self.copies_in_video_memory() {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
        } else {
            heap
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

impl Host {
    /// An unmapped `D3D12_HEAP_TYPE_READBACK` buffer of `size` bytes in
    /// `D3D12_RESOURCE_STATE_COPY_DEST`.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }

    /// A mapped buffer of `size` bytes: upload memory, or (`readback`)
    /// readback memory in the copy-destination state.
    fn mapped_buffer(&self, size: u64, readback: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback { D3D12_HEAP_TYPE_READBACK } else { D3D12_HEAP_TYPE_UPLOAD },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    if readback { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, readback, "no mapped buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    fn ready(&self, now: Instant) -> bool {
        self.next_attempt.is_none_or(|next| now >= next)
    }

    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if !self.ready(now) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    /// The queue the swap chain presents on.
    queue: ID3D12CommandQueue,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    /// The overlay blend pass, once tried (#15).
    compositor: Option<std::result::Result<crate::composite::Compositor, ()>>,
    /// The texture the blend pass writes.
    blended: Option<ID3D12Resource>,
}

#[derive(Clone, Copy)]
struct FrameWindow {
    since: Instant,
    frames: u64,
    drawn: u64,
}

struct FrameRates {
    frames: f64,
    fps: f64,
    ms: f64,
    draws: f64,
}

impl FrameWindow {
    /// Advances this window once its own interval has elapsed.
    fn sample(&mut self, next: Self, interval: Duration) -> Option<FrameRates> {
        let elapsed = next.since.duration_since(self.since);
        if elapsed < interval {
            return None;
        }
        let seconds = elapsed.as_secs_f64();
        let frames = (next.frames - self.frames) as f64;
        let draws = (next.drawn - self.drawn) as f64 / frames;
        *self = next;
        Some(FrameRates {
            frames,
            fps: frames / seconds,
            ms: seconds * 1000.0 / frames,
            draws,
        })
    }
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    frames: u64,
    caption_window: FrameWindow,
    report_window: FrameWindow,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let start = FrameWindow {
            since: Instant::now(),
            frames: 0,
            drawn: crate::draw::drawn(),
        };
        Meter {
            frames: 0,
            caption_window: start,
            report_window: start,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        let next = FrameWindow {
            since: Instant::now(),
            frames: self.frames,
            drawn: crate::draw::drawn(),
        };
        let Some(caption_rates) = self.caption_window.sample(next, Duration::from_secs(1)) else {
            return;
        };
        if let Some(report) = self.report_window.sample(next, Duration::from_secs(5)) {
            let FrameRates { frames: report_frames, fps, ms, draws } = report;
            let [cpu, gpu] = SUBMISSION_US.each_ref().map(|counter| {
                counter.swap(0, Ordering::Relaxed) as f64 / 1000.0 / report_frames
            });
            let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
                let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
                PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / report_frames
            });
            let [back_ms, back_mb] = [0, 1].map(|i| {
                COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / report_frames
            });
            let dropped = crate::draw::take_dropped() as f64 / report_frames;
            let indirect = crate::list::take_indirect() as f64 / report_frames;
            let root_views = crate::draw::take_root_view_slots() as f64 / report_frames;
            let [records, counted] =
                crate::indirect::take_records().map(|n| n as f64 / report_frames);
            let [hoisted, inline] =
                crate::indirect::take_clamps().map(|n| n as f64 / report_frames);
            let [anywhere, theirs, after_inline, unbatched] =
                crate::indirect::take_inline_why().map(|n| n as f64 / report_frames);
            let clamp_cuts = crate::indirect::take_cuts() as f64 / report_frames;
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame; indirect \
                 max records {records:.0}/frame ({counted:.0} under a count; clamps \
                 {hoisted:.0} hoisted, {inline:.0} inline: {anywhere:.0} after a write \
                 anywhere, {theirs:.0} after a write of theirs, {after_inline:.0} after an \
                 inline one, {unbatched:.0} unbatched; {clamp_cuts:.1} host lists started \
                 for clamps)",
                making = crate::pipeline::making()
            );
            crate::ray_skip::log(report_frames);
            crate::ambient_estimate::log(report_frames);
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let (heaps, heap_used, heap_free) = crate::placement::stats();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 heaps of textures {heaps} ({:.0} MB used, {:.0} MB free, {} emptied giving up {:.0} MB); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                heap_used as f64 / 1e6,
                heap_free as f64 / 1e6,
                crate::resource::HEAPS_EMPTIED[0].load(Ordering::Relaxed),
                crate::resource::HEAPS_EMPTIED[1].load(Ordering::Relaxed) as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
            );
            crate::twin::log(report_frames);
            crate::census::log(report_frames);
            crate::draw::table_memo_line(report_frames);
            // How long the threads that ran held work were held by it: a
            // title thread waits there instead of going on with its frame.
            let ms = |us: u64| us as f64 / 1e3 / report_frames;
            let held: Vec<String> = crate::queue::WHY_NAMES
                .iter()
                .zip(crate::queue::take_blocked())
                .map(|(why, [lock, run])| format!("{why} {:.1} + {:.1}", ms(lock), ms(run)))
                .collect();
            tracing::info!(
                "threads held by host GPU work per frame, ms waiting for the queues' lock + \
                 running it: {}",
                held.join(", ")
            );
            // With VICTORY_ASYNC_SUBMIT=1 / VICTORY_FLIGHT1=1: what the
            // submit worker did instead, and how long the host GPU waited.
            flight::log_frame(report_frames);
            crate::queue::log_drains(report_frames);
            crate::queue::log_title_gap();
            let phases: Vec<String> = PHASE_NAMES
                .iter()
                .zip(take_phases())
                .map(|(name, us)| format!("{name} {:.1}", ms(us)))
                .collect();
            tracing::info!("host preparing per frame by phase, ms: {}", phases.join(", "));
            {
                let n = STEPS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
                let kinds = STEP_NAMES.len();
                let parts: Vec<String> = STEP_NAMES
                    .iter()
                    .enumerate()
                    .map(|(k, name)| {
                        if k == 1 || k == 2 {
                            format!("{name} {:.0}", n[k] as f64 / report_frames)
                        } else {
                            format!("{name} {:.0} ({:.1} ms)", n[k] as f64 / report_frames, ms(n[kinds + k]))
                        }
                    })
                    .collect();
                tracing::info!(
                    "host list steps per frame (the rest of the phase is the refreshes and slots): {}",
                    parts.join(", ")
                );
            }
            #[cfg(windows)]
            crate::descriptor::log_sync(report_frames);
            crate::shadow::log_refreshed(report_frames);
            crate::alias::log_unaliased();
            {
                let k = COPY_KINDS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
                let per = |n: u64| n as f64 / report_frames;
                let mb = |b: u64| b as f64 / 1e6 / report_frames;
                let [compared, compared_bytes, same, same_bytes, comparing] = same_copy::take();
                // The refreshes are the rest of the copies' line (both
                // counters were taken above, as ms and MB per frame).
                let refresh_ms = (copy_ms - ms(k[1] + k[4])).max(0.0);
                let refresh_mb = (copy_mb - mb(k[2] + k[5])).max(0.0);
                tracing::info!(
                    "host preparing copies per frame: title copies {:.1} ({:.1} MB, {:.1} ms), fills of upload \
                     memory {:.1} ({:.1} MB, {:.1} ms), refreshes of host copies {refresh_mb:.1} MB \
                     {refresh_ms:.1} ms; title copies compared {:.1} ({:.1} MB, {:.1} ms), already in place \
                     {:.1} ({:.1} MB){}",
                    per(k[0]),
                    mb(k[2]),
                    ms(k[1]),
                    per(k[3]),
                    mb(k[5]),
                    ms(k[4]),
                    per(compared),
                    mb(compared_bytes),
                    ms(comparing),
                    per(same),
                    mb(same_bytes),
                    if same_copy::mode() == same_copy::Mode::Skip { ", left out" } else { "" },
                );
            }
            #[cfg(feature = "diagnostics")]
            {
                if let Some([(local, budget), (system, system_budget)]) = video_memory() {
                    let (commit, working_set) = process_memory();
                    let [views_total, views_live, views_cbv, views_placed] =
                        crate::descriptor::views_kept();
                    tracing::info!(
                        "video memory: {local:.0} MB of a {budget:.0} MB budget on the card, \
                         {system:.0} MB of {system_budget:.0} MB in system memory; process commit {:.0} MB, \
                         working set {:.0} MB; Rust heap {:.0} MB; placed textures made {:.0} MB; \
                         views {views_live}/{views_total} (cbv {views_cbv}, placed {views_placed}); \
                         list upload memory {} blocks {:.0} MB",
                        commit as f64 / 1e6,
                        working_set as f64 / 1e6,
                        crate::copystat::HEAP_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::resource::PLACED_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::list::UPLOADS[0].load(Ordering::Relaxed),
                        crate::list::UPLOADS[1].load(Ordering::Relaxed) as f64 / 1e6,
                    );
                }
                crate::shadow::log_held();
                let [gpu, title] = crate::resource::texture_census().map(|r| r.map(|b| b as f64 / 1e6));
                tracing::info!(
                    "host texture MB (used by lists, viewed by heap slots only, neither): GPU-written {:.0} {:.0} {:.0}, \
                     from title memory {:.0} {:.0} {:.0}",
                    gpu[0], gpu[1], gpu[2], title[0], title[1], title[2]
                );
                let [given, given_bytes, remade, soon] =
                    [0, 1, 2, 3].map(|i| crate::resource::EVICTIONS[i].swap(0, Ordering::Relaxed));
                tracing::info!(
                    "host textures given up: {given} ({:.0} MB); made again after being given up: {remade}, {soon} of them within {} frames",
                    given_bytes as f64 / 1e6,
                    crate::resource::SOON_FRAMES,
                );
                // Window caption updates leave the report counters intact.
                let drops = crate::pipeline::take_drops();
                crate::frame::log();
                log_gpu_times(report_frames);
                log_gpu_walls(report_frames);
                crate::copystat::log(report_frames);
                crate::bindless::log(report_frames);
                crate::heap_worker::log(report_frames);
                crate::shadow::log_views(report_frames);
                crate::shadow::log_lookups(report_frames);
                let [n, bytes] = [0, 1].map(|i| crate::list::READ_BACKS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "texture read backs: {:.1}/frame, {:.1} KB/frame",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3
                    );
                }
                let [n, bytes, own] =
                    [0, 1, 2].map(|i| crate::list::FOOTPRINT_UPLOADS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "buffer to texture copies: {:.1}/frame, {:.1} KB/frame, {:.1}/frame in a buffer of their own",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3,
                        own as f64 / report_frames
                    );
                }
                #[cfg(feature = "diagnostics")]
                {
                    let (lists, bytes) = crate::list::spare_scratch();
                    let shed = crate::list::SPARE_SHED.swap(0, Ordering::Relaxed);
                    tracing::info!(
                        "spare host lists: {lists}, keeping {:.1} MB of scratch; {:.1} MB given up since the last line",
                        bytes as f64 / 1e6,
                        shed as f64 / 1e6
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / report_frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let FrameRates { fps, ms, draws, .. } = caption_rates;
        let text = format!("{fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = crate::caption::with_stats(caption, &text);
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on), with `overlay` (a
    /// texture and its host state, which it keeps) blended over it when
    /// `VICTORY_PRESENT_COMPOSITE=1`. `None` if nothing was shown; else, in
    /// diagnostics builds, whether the frame was black.
    pub(crate) fn present(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(picture_format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        // The overlay with its view format, when it is to be blended in.
        let overlay = overlay.filter(|_| crate::composite::wanted()).and_then(|(o, s)| {
            // SAFETY: a live resource.
            let f = swap_format(unsafe { o.GetDesc() }.Format)?;
            Some((o, s, f))
        });
        let native = (desc.Width as u32, desc.Height);
        let size = crate::composite::output_size(native.0, native.1);
        let can_sample = desc.SampleDesc.Count == 1
            && desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE == D3D12_RESOURCE_FLAG_NONE
            && (desc.Format == picture_format || crate::resource::family(desc.Format.0 as u32) == desc.Format.0 as u32);
        let use_pass = overlay.is_some() || (size != native && can_sample);
        let (width, height) = if use_pass { size } else { native };
        let format = if use_pass { crate::composite::result_format(picture_format) } else { picture_format };
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        // A swap chain presents on the queue it was made with. Once that
        // queue stopped running work and was replaced ([`Host::lost`]), the
        // chain goes first (a window takes one at a time), and a new one is
        // made for the same window on the new queue.
        let queue = self.queue();
        let mut shown = None;
        if presenter.as_ref().is_some_and(|p| p.queue != queue) {
            shown = presenter.take().map(|p| p.hwnd);
        }
        if presenter.is_none() {
            *presenter = match retry.attempt(Instant::now(), || {
                self.presenter(&queue, shown, width, height, format)
            })? {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        // The blend pass and the texture it writes, made on first use.
        let mut blended = None;
        if use_pass {
            if p.compositor.is_none() {
                p.compositor = Some(crate::composite::Compositor::new(&self.device).map_err(|e| {
                    tracing::warn!(error = %e, "no overlay blending: the pass did not compile");
                }));
            }
            if let Some(Ok(compositor)) = &p.compositor {
                let current = p.blended.as_ref().is_some_and(|t| {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    (d.Width, d.Height, d.Format) == (u64::from(width), height, format)
                });
                if !current {
                    p.blended =
                        crate::composite::result_texture(&self.device, u64::from(width), height, format).ok();
                }
                if let Some(result) = &p.blended {
                    blended = Some((compositor, result.clone()));
                }
            }
        }
        // If the pass or its output is unavailable, the back buffer must
        // match the original texture before the direct CopyResource below.
        let (width, height, format) = if blended.is_some() {
            (width, height, format)
        } else {
            (native.0, native.1, picture_format)
        };
        // What is presented, whenever it changes: the title's texture in its
        // host format, and the swap chain format it is copied into.
        /// Width, height and format.
        type Shape = (u32, u32, i32);
        static SHOWN: Mutex<Option<(Shape, Shape)>> = Mutex::new(None);
        let shown = Some(((native.0, native.1, desc.Format.0), (width, height, format.0)));
        let mut last = SHOWN.lock().unwrap_or_else(|e| e.into_inner());
        if *last != shown {
            *last = shown;
            tracing::info!(
                frame,
                width = desc.Width,
                height = desc.Height,
                texture_format = desc.Format.0,
                flags = desc.Flags.0,
                swap_format = format.0,
                output_width = width,
                output_height = height,
                "presenting a title texture"
            );
        }
        drop(last);
        // This also covers a swap chain created at the capped size when the
        // blend pass later cannot be built: width, height and format were reset
        // to the title texture's own above, so the buffers are resized to match
        // before the direct copy.
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // What is copied to the back buffer: the blend or the picture,
            // and the state it returns to afterwards.
            let (source, state) = match &blended {
                Some((compositor, result)) => {
                    list.ResourceBarrier(&[transition(texture, state, READ)]);
                    if let Some((o, ostate, _)) = overlay {
                        list.ResourceBarrier(&[transition(o, D3D12_RESOURCE_STATES(ostate as i32), READ)]);
                    }
                    compositor.record(&self.device, list, (texture, picture_format),
                        overlay.map(|(o, _, f)| (o, f)), result, format);
                    if let Some((o, ostate, _)) = overlay {
                        list.ResourceBarrier(&[transition(o, READ, D3D12_RESOURCE_STATES(ostate as i32))]);
                    }
                    list.ResourceBarrier(&[
                        transition(texture, READ, state),
                        transition(result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    ]);
                    (result.clone(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                }
                None => {
                    list.ResourceBarrier(&[transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    (texture.clone(), state)
                }
            };
            let texture = &source;
            list.ResourceBarrier(&[transition(
                &back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            // With work queued for the submit worker, PresentX drained it just
            // before (`queue::present`): the copy goes behind the host
            // submission in flight only, ahead of what other threads queued
            // since (the next frame's, as on the console), rather than
            // draining again behind all of it.
            if async_submit_enabled() || flight1_enabled() {
                self.execute_sync(&[&p.copy], &[]);
            } else {
                self.execute(&[&p.copy]);
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    /// A swap chain on `queue` for `shown`, else for the title's window,
    /// else for a window of our own.
    fn presenter(
        &self,
        queue: &ID3D12CommandQueue,
        shown: Option<HWND>,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match shown.or_else(title_window) {
            Some(hwnd) => hwnd,
            None if std::env::var("VICTORY_WINDOW_SECONDARY").is_ok_and(|v| {
                v == "1"
                    || v.split_once(',').is_some_and(|(x, y)| {
                        x.trim().parse::<i32>().is_ok() && y.trim().parse::<i32>().is_ok()
                    })
            }) =>
            {
                return Err(("secondary title window unavailable", windows::core::Error::empty()));
            }
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        self.presenter_for_window(queue, hwnd, width, height, format)
    }

    fn presenter_for_window(
        &self,
        queue: &ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                queue: queue.clone(),
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                compositor: None,
                blended: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// Diagnostics, `VICTORY_COPIES_IN_VIDEO_MEMORY`: [`HostMemory::shared_buffer`]s
    /// (host copies of title memory among them) in video memory the CPU maps
    /// (a GPU upload heap, where the host has one), to measure how much of
    /// the GPU's time is shaders reading the copies over the bus. CPU reads
    /// of them (snapshots, copy-backs) become slow: for that measurement only.
    #[cfg(feature = "diagnostics")]
    fn copies_in_video_memory(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_COPIES_IN_VIDEO_MEMORY") {
                return false;
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            }
            .is_ok()
                && options.GPUUploadHeapSupported.as_bool();
            tracing::warn!(on = ok, "host copies of title memory in video memory (diagnostics)");
            ok
        })
    }

    /// 256-byte-aligned GPU VA of a shared zero-filled 64 KiB buffer bound
    /// in place of null, unset, or unmapped direct root CBVs and SRVs.
    pub(crate) fn zero_root_read(&self) -> Option<u64> {
        self.zero_root_read
            .get_or_init(|| {
                let bytes = (crate::watch::PAGE + 256) as u64;
                let (buffer, data) = self.shared_buffer(bytes)?;
                name(&buffer, || "null root read".into());
                // SAFETY: `data` points to `bytes` mapped bytes; `buffer` is live.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress().next_multiple_of(256)
                };
                Some((buffer, gpu))
            })
            .as_ref()
            .map(|&(_, gpu)| gpu)
    }

    /// 256-byte-aligned GPU VA of an isolated 64 KiB throwaway buffer bound
    /// in place of null, unset, or unmapped direct root UAVs so shader writes
    /// never fault or corrupt [`Host::zero_root_read`].
    pub(crate) fn sink_root_uav(&self) -> Option<u64> {
        self.sink_root_uav
            .get_or_init(|| {
                let bytes = (crate::watch::PAGE + 256) as u64;
                let (buffer, data) = self.shared_buffer(bytes)?;
                name(&buffer, || "null root uav sink".into());
                // SAFETY: `data` points to `bytes` mapped bytes; `buffer` is live.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress().next_multiple_of(256)
                };
                Some((buffer, gpu))
            })
            .as_ref()
            .map(|&(_, gpu)| gpu)
    }
}

/// Completed host submission count (`Host::execute`).
pub(super) static SUBMISSIONS_DONE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Number of completed host submissions so far.
pub(crate) fn submissions_done() -> u64 {
    SUBMISSIONS_DONE.load(Ordering::Acquire)
}

/// Host submission count when the latest one was handed to the queue
/// (`Host::execute`, Flight-1 and the queue check): stored before its lists
/// go to the queue, under `Host::submitted`.
pub(super) static SUBMISSIONS_BEGUN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Number of host submissions handed to the queue so far. While it is
/// ahead of [`submissions_done`], the GPU may still run the newest ones.
pub(crate) fn submissions_begun() -> u64 {
    SUBMISSIONS_BEGUN.load(Ordering::Acquire)
}

/// How many host submissions a retired GPU object or recycled command list
/// stays alive after the submission current when it was retired (`16` ~= 2
/// frames; `VICTORY_BURY_LAG=0` drops right after the current submission).
pub(crate) fn bury_lag() -> u64 {
    #[cfg(test)]
    if let Some(lag) = TEST_BURY_LAG.with(std::cell::Cell::get) {
        return lag;
    }
    static LAG: OnceLock<u64> = OnceLock::new();
    *LAG.get_or_init(|| {
        std::env::var("VICTORY_BURY_LAG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

#[cfg(test)]
thread_local! {
    static TEST_BURY_LAG: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_bury_lag<R>(lag: u64, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<u64>);
    impl Drop for Guard {
        fn drop(&mut self) {
            TEST_BURY_LAG.with(|cell| cell.set(self.0));
        }
    }
    let prev = TEST_BURY_LAG.with(|cell| cell.replace(Some(lag)));
    let _guard = Guard(prev);
    f()
}

struct Retired(#[allow(dead_code)] Box<dyn std::any::Any + Send>);

/// Host GPU objects (`ID3D12Resource`, `ID3D12Heap`, submission batches) kept
/// alive across [`bury_lag`] submissions so D3DMetal/AGX never faults on
/// recently retired virtual addresses.
static RETIRED: Mutex<std::collections::VecDeque<(u64, Retired)>> =
    Mutex::new(std::collections::VecDeque::new());

/// Keeps `resource` alive until [`bury_lag`] host submissions after now.
pub(crate) fn retire(resource: ID3D12Resource) {
    retire_any(resource);
}

/// Keeps `object` alive until [`bury_lag`] host submissions after now, then
/// drops it on the background release thread.
pub(crate) fn retire_any<T: Send + 'static>(object: T) {
    let at = submissions_done();
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((at, Retired(Box::new(object))));
}

/// Releases retired objects whose [`bury_lag`] window has elapsed at
/// `done_submission`.
fn advance_retired(done_submission: u64) {
    let lag = bury_lag();
    let ready: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        let n = retired
            .iter()
            .take_while(|(at, _)| at.saturating_add(lag) <= done_submission)
            .count();
        if n == 0 {
            return;
        }
        retired.drain(..n).collect()
    };
    release_later(Box::new(ready));
}

/// Batched [`retire`] under one hold of the retired-resources lock.
pub(crate) fn retire_all(resources: Vec<ID3D12Resource>) {
    if resources.is_empty() {
        return;
    }
    let at = submissions_done();
    let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
    retired.extend(resources.into_iter().map(|r| (at, Retired(Box::new(r)))));
}

/// Immediately releases all retired objects (tests).
#[cfg(test)]
pub(crate) fn release_retired() {
    let retired: Vec<(u64, Retired)> = RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .drain(..)
        .collect();
    if !retired.is_empty() {
        release_later(Box::new(retired));
    }
}

/// Drops `what` on a thread of its own: releasing a D3D12 resource is a
/// kernel call (~5 % of a world frame on the title's submitting thread,
/// run perf1). Only for objects no GPU work in flight uses.
fn release_later(what: Box<dyn std::any::Any + Send>) {
    type Release = std::sync::mpsc::Sender<Box<dyn std::any::Any + Send>>;
    static RELEASE: OnceLock<Release> = OnceLock::new();
    let release = RELEASE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn std::any::Any + Send>>();
        // Without the thread the receiver is gone: sends fail and drop here.
        let _ = std::thread::Builder::new()
            .name("victory-release".into())
            .spawn(move || rx.into_iter().for_each(drop));
        tx
    });
    let _ = release.send(what);
}

/// Waits until the release thread has dropped what was sent to it so far
/// (tests that check what a finished submission freed: [`retire_any`]
/// drops it there, after the submission returns).
#[cfg(test)]
pub(crate) fn wait_for_releases() {
    struct Mark(std::sync::mpsc::Sender<()>);
    impl Drop for Mark {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    release_later(Box::new(Mark(tx)));
    let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds: all of
/// them in listed capture frames, the first 500 elsewhere.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if is_traced(what)
        && (crate::capture::listed(crate::queue::presented() + 1)
            || LOGGED.fetch_add(1, Ordering::Relaxed) < 500)
    {
        tracing::info!("traced command: {what}");
    }
}

/// The shader hashes `VICTORY_TRACE_SHADERS` lists (comma-separated;
/// diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_shaders() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// Whether `what` names a shader [`traced_shaders`] lists.
#[cfg(feature = "diagnostics")]
pub(crate) fn is_traced(what: &str) -> bool {
    traced_shaders().iter().any(|h| what.contains(h.as_str()))
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!(
                    "; list {i} of {}, of its {}: {n}",
                    runs.len(),
                    notes.len()
                ));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(
    history: &[D3D12_AUTO_BREADCRUMB_OP],
    markers: &[(u32, String)],
    done: usize,
) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done]
        .iter()
        .rposition(barrier)
        .map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::ISSUED.with(|l| l.borrow_mut().push((resource.as_raw() as usize, before.0 as u32, after.0 as u32)));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A barrier that waits for every unordered-access write so far.
pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// Records a single Enhanced `D3D12_GLOBAL_BARRIER` on `cmd7`.
///
/// D3DMetal `MPL` does not track buffers bound via root descriptors
/// (`SetComputeRootShaderResourceView`/`SetComputeRootUnorderedAccessView`) or
/// bindless heaps in per-`MTLBuffer` barrier maps; a global barrier routes
/// straight to `MPLCommandList::GlobalBarrierImpl(beforeStage, afterStage)` and
/// honors exact `MPLStage` masks.
pub(crate) unsafe fn global_barrier(
    cmd7: &ID3D12GraphicsCommandList7,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
) {
    let barrier = D3D12_GLOBAL_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
    };
    let group = D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_GLOBAL,
        NumBarriers: 1,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pGlobalBarriers: &barrier,
        },
    };
    // SAFETY: caller guarantees `cmd7` is an open command list; `barrier` lives
    // across the synchronous `Barrier` call.
    unsafe {
        cmd7.Barrier(&[group]);
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {
    // `mod tests` is already taken at the top of this file
    use super::Batch;
    use crate::backend::HostGpu;
    use crate::shadow::Shadow;
    use crate::watch::PAGE;
    use std::sync::Arc;


    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A list asking again for what the submission already brought up to
    /// date copies nothing, and the copystat line says so: the skipped
    /// bytes are what staying current saved the later lists.
    #[test]
    #[cfg(feature = "diagnostics")]
    fn a_still_current_ask_is_counted_skipped() {
        let Some((p, copy)) = plain(2) else {
            return;
        };
        let len = 2 * PAGE;
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert!(batch.flush() > 0, "all of it at first");
        let (skips, bytes) = (
            crate::copystat::count(crate::copystat::C::Skipped),
            crate::copystat::count(crate::copystat::C::SkippedBytes),
        );
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        assert!(
            crate::copystat::count(crate::copystat::C::Skipped) > skips,
            "the ask counted"
        );
        assert!(
            crate::copystat::count(crate::copystat::C::SkippedBytes) >= bytes + len as u64,
            "its bytes"
        );
        done(p, Some(copy));
    }

    /// A copy of `pages` pages of plain private memory, which is not the
    /// runtime's: untracked, copied whole whatever was asked for. None
    /// without a host GPU.
    fn plain(pages: usize) -> Option<(usize, Arc<Shadow>)> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        // SAFETY: a fresh allocation, freed by `done`.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0);
        crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
        let copy = crate::shadow::find(p as u64, (pages * PAGE) as u64);
        if copy.is_none() {
            done(p, None);
        }
        copy.map(|c| (p, c))
    }

    fn done(p: usize, copy: Option<Arc<Shadow>>) {
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// Fills `len` bytes of title memory at `at`.
    fn fill(at: usize, len: usize, byte: u8) {
        // SAFETY: memory of the test's own allocations.
        unsafe { std::ptr::write_bytes(at as *mut u8, byte, len) };
    }

    /// A title copy of `len` bytes from `src` to `dst`, as a list's step
    /// runs it.
    fn title_copy(batch: &mut Batch, dst: usize, src: usize, len: usize) {
        batch.title_copied(dst, len);
        super::copy_mapped(dst as *mut u8, src as *const u8, len, true);
    }

    /// Runs a list holding one title copy of `len` bytes from `src` to
    /// `dst` with `VICTORY_SAME_COPIES` as `mode` on this thread (1 probe,
    /// 2 skip).
    fn copy_with(batch: &mut Batch, mode: u8, dst: usize, src: usize, len: usize) {
        let list = super::host().unwrap().new_list().unwrap();
        list.defer_copy(dst as *mut u8, src as *const u8, len);
        super::same_copy::TEST_MODE.with(|m| m.set(mode));
        list.run_deferred(batch);
        super::same_copy::TEST_MODE.with(|m| m.set(0));
    }

    /// The list-steps line counts each step, and times host work.
    #[test]
    fn list_steps_are_counted_by_kind() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let list = host.new_list().unwrap();
        list.defer_host(|| std::thread::sleep(std::time::Duration::from_millis(2)));
        let before = super::STEPS[3].load(std::sync::atomic::Ordering::Relaxed);
        let before_us = super::STEPS[super::STEP_NAMES.len() + 3].load(std::sync::atomic::Ordering::Relaxed);
        list.run_deferred(&mut Batch::default());
        // Another test's frame report may take the counts meanwhile.
        let (after, after_us) = (
            super::STEPS[3].load(std::sync::atomic::Ordering::Relaxed),
            super::STEPS[super::STEP_NAMES.len() + 3].load(std::sync::atomic::Ordering::Relaxed),
        );
        assert!(after > before || after >= 1, "{before} then {after}");
        assert!(after_us >= before_us + 2000 || after_us >= 2000, "{before_us} then {after_us} us");
    }

    /// The frame report's refreshed-copies line counts each copy's
    /// refreshed bytes, once per range brought up.
    #[test]
    fn refreshes_are_counted_by_copy() {
        let Some((p, a)) = plain(2) else {
            return; // no host GPU here
        };
        fill(p, 2 * PAGE, 0x5a);
        let before = crate::shadow::refreshed_bytes(&a);
        let mut batch = Batch::default();
        batch.add(&a, p, PAGE);
        assert_eq!(batch.flush(), PAGE);
        // Another test's frame report may take the counts meanwhile.
        let after = crate::shadow::refreshed_bytes(&a);
        assert!(after == before + PAGE as u64 || after == PAGE as u64, "{before} then {after}");
        done(p, Some(a));
    }

    /// `VICTORY_SAME_COPIES=skip`: a title copy whose bytes the destination
    /// and its host copy hold already is left out, so the copy stays
    /// current and nothing is refreshed again; with `probe` it still runs.
    #[test]
    fn a_title_copy_of_bytes_in_place_is_left_out() {
        // Memory another test gave back may still show elsewhere or have a
        // second copy registered: then nothing is compared (as it should
        // not be). Such memory is kept, and other memory taken.
        let mut kept = Vec::new();
        let (p, a) = loop {
            let Some((p, a)) = plain(2) else {
                return; // no host GPU here
            };
            if !crate::watch::aliased(p, 2 * PAGE) && crate::shadow::over(p, p + 2 * PAGE).len() == 1 {
                break (p, a);
            }
            assert!(kept.len() < 8, "no plain memory without other copies");
            kept.push(a);
        };
        fill(p, 2 * PAGE, 0x5a);
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * PAGE);
        assert_eq!(batch.flush(), 2 * PAGE, "the host copy takes the title's bytes");

        copy_with(&mut batch, 2, p + PAGE, p, 256);
        batch.add(&a, p, 2 * PAGE);
        assert_eq!(batch.flush(), 0, "left out: the copy is still current");

        copy_with(&mut batch, 1, p + PAGE, p, 256);
        batch.add(&a, p, 2 * PAGE);
        assert!(batch.flush() > 0, "probe: the copy ran and its destination is due again");
        done(p, Some(a));
    }

    /// What comparing a title copy costs against running it, under Wine:
    /// `cargo test --release -p victory-gpu --lib same_copy_costs --
    /// --ignored --nocapture` (through `tools/macos/test-wine.sh`).
    #[test]
    #[ignore = "times title copies against comparing them"]
    fn same_copy_costs() {
        let pages = 2 * (8 << 20) / PAGE;
        let Some((p, a)) = plain(pages) else {
            return; // no host GPU here
        };
        let half = pages / 2 * PAGE;
        fill(p, 2 * half, 0x5a);
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * half);
        batch.flush();
        for len in [4 << 10, 64 << 10, 1 << 20, 8 << 20] {
            let rounds = (256 << 20) / len;
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                super::copy_mapped((p + half) as *mut u8, p as *const u8, len, true);
            }
            let copy = started.elapsed().as_secs_f64() / rounds as f64;
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                assert!(super::same_copy::in_place(p + half, p, len));
            }
            let compare = started.elapsed().as_secs_f64() / rounds as f64;
            println!(
                "{:>5} KiB: copy {:.1} us ({:.1} GB/s), compare {:.1} us ({:.1} GB/s)",
                len >> 10,
                copy * 1e6,
                len as f64 / copy / 1e9,
                compare * 1e6,
                len as f64 / compare / 1e9
            );
        }
        done(p, Some(a));
    }

    /// A title copy that changes the destination, or whose destination's
    /// host copy holds other bytes than the title's, runs with `skip` too.
    #[test]
    fn a_title_copy_that_changes_anything_still_runs() {
        let Some((p, a)) = plain(2) else {
            return; // no host GPU here
        };
        fill(p, 2 * PAGE, 0x5a);
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * PAGE);
        batch.flush();

        // Other bytes at the source.
        fill(p, 256, 0x11);
        copy_with(&mut batch, 2, p + PAGE, p, 256);
        // SAFETY: committed title memory.
        assert_eq!(unsafe { *((p + PAGE) as *const u8) }, 0x11, "the copy ran");
        batch.add(&a, p, 2 * PAGE);
        assert!(batch.flush() > 0);
        assert_eq!(a.read_host((p + PAGE) as u64, 1)[0], 0x1111_1111);

        // The title's source and destination agree, but the host copy holds
        // other bytes there that nothing makes due (what shaders wrote and
        // kept in the copy): the copy runs, as on the console it would
        // overwrite them, and the refresh it makes due puts the source's
        // bytes into the host copy.
        fill(p, 256, 0x77);
        fill(p + PAGE, 256, 0x77);
        batch.add(&a, p, 2 * PAGE);
        batch.flush();
        a.write_host((p + PAGE) as u64, &[0xee; 256]);
        copy_with(&mut batch, 2, p + PAGE, p, 256);
        batch.add(&a, p, 2 * PAGE);
        assert!(batch.flush() > 0, "the copy made its destination due again");
        assert_eq!(a.read_host((p + PAGE) as u64, 1)[0], 0x7777_7777);
        done(p, Some(a));
    }

    #[test]
    fn overlapping_title_copy_refreshes_its_destination() {
        let Some((pa, a)) = plain(4) else {
            return; // no host GPU here
        };
        let (pb, b) = plain(4).expect("second host buffer");
        fill(pa, PAGE, 0x12);
        fill(pa + PAGE, PAGE, 0x34);
        fill(pb, 4 * PAGE, 0x56);
        let mut batch = Batch::default();
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        let list = super::host().unwrap().new_list().unwrap();
        list.defer_move((pa + PAGE) as *mut u8, pa as *const u8, 2 * PAGE);
        list.run_deferred(&mut batch);
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        assert_eq!(
            batch.flush(),
            2 * PAGE + 4 * PAGE,
            "the overlapping copy's destination plus the disjoint copy deferred until flush"
        );
        assert_eq!(a.read_host((pa + PAGE) as u64, 1)[0], 0x1212_1212);
        assert_eq!(a.read_host((pa + 2 * PAGE) as u64, 1)[0], 0x3434_3434);
        assert_eq!(b.read_host(pb as u64, 1)[0], 0x5656_5656);
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        assert_eq!(batch.flush(), 0, "both copies are now current");
        done(pa, Some(a));
        done(pb, Some(b));
    }

    /// A non-aliased title copy flushes only overlapping pending refreshes,
    /// invalidates their destination coverage, and leaves disjoint pending
    /// shadows queued so subsequent `add`s coalesce until final `flush()`.
    #[test]
    fn selective_title_copy_flushes_only_overlapping_pending_shadows() {
        let Some((pa, a)) = plain(4) else {
            return; // no host GPU here
        };
        let (pb, b) = plain(4).expect("second host buffer");
        let (pc, c) = plain(2).expect("third host buffer");
        fill(pa, 4 * PAGE, 0x11);
        fill(pb, 4 * PAGE, 0x22);
        fill(pc, 2 * PAGE, 0x33);

        let mut batch = Batch::default();
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        batch.add(&c, pc, 2 * PAGE);
        assert_eq!(batch.refreshes.len(), 3);
        assert_eq!(batch.pending.len(), 3);

        // Copy into `a` (overlapping) while `b` and `c` are disjoint.
        let src = [0xAAu8; 64];
        title_copy(&mut batch, pa + PAGE, src.as_ptr() as usize, src.len());

        // `a` was flushed before the copy, while `b` and `c` stayed queued.
        assert_eq!(
            batch.refreshes.len(),
            2,
            "disjoint shadows remain queued across title_copied"
        );
        assert_eq!(batch.pending.len(), 2);
        assert_eq!(
            a.read_host(pa as u64, 1)[0],
            0x1111_1111,
            "overlapping shadow was flushed before title copy"
        );
        assert_eq!(
            b.read_host(pb as u64, 1)[0],
            0,
            "disjoint shadow was not prematurely flushed"
        );
        assert_eq!(
            c.read_host(pc as u64, 1)[0],
            0,
            "second disjoint shadow was not prematurely flushed"
        );

        // Subsequent adds for `b` and `c` coalesce into their still-pending
        // entries; `a` queues only the invalidated destination window.
        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        batch.add(&c, pc, 2 * PAGE);
        assert_eq!(batch.refreshes.len(), 3);
        assert_eq!(batch.pending.len(), 3);

        let dest_len = if crate::watch::fault_tracking() {
            PAGE
        } else {
            src.len()
        };
        assert_eq!(
            batch.flush(),
            dest_len + 4 * PAGE + 2 * PAGE,
            "final flush copies only `a`'s destination plus the two deferred disjoint shadows once"
        );
        assert_eq!(a.read_host((pa + PAGE) as u64, 1)[0], 0xAAAA_AAAA);
        assert_eq!(b.read_host(pb as u64, 1)[0], 0x2222_2222);
        assert_eq!(c.read_host(pc as u64, 1)[0], 0x3333_3333);

        batch.add(&a, pa, 4 * PAGE);
        batch.add(&b, pb, 4 * PAGE);
        batch.add(&c, pc, 2 * PAGE);
        assert_eq!(batch.flush(), 0, "all three shadows are current after final flush");

        done(pa, Some(a));
        done(pb, Some(b));
        done(pc, Some(c));
    }

    /// An aliased title copy still falls back to flushing every pending
    /// shadow conservatively, even disjoint ones.
    #[test]
    fn aliased_title_copy_still_flushes_pending_disjoint_shadows() {
        let (a, _b) = crate::watch::tests::twin_views(2);
        let Some((pd, d)) = plain(2) else {
            return; // no host GPU here
        };
        fill(pd, 2 * PAGE, 0x44);
        let src = crate::watch::tests::pool(1);
        fill(src, PAGE, 0x55);

        let mut batch = Batch::default();
        batch.add(&d, pd, 2 * PAGE);
        assert_eq!(batch.refreshes.len(), 1);
        title_copy(&mut batch, a, src, PAGE);
        assert_eq!(
            batch.refreshes.len(),
            0,
            "aliased copy must conservatively flush pending disjoint shadows"
        );
        assert_eq!(d.read_host(pd as u64, 1)[0], 0x4444_4444);

        done(pd, Some(d));
        done(src, None);
    }

    /// Two copies of title memory, every list of a submission asking for
    /// both whole between title copies: a title copy makes due again only
    /// the bytes it wrote, and copies far apart in one copy are two
    /// ranges, not what is between them.
    #[test]
    fn a_title_copy_makes_due_again_only_the_bytes_it_wrote() {
        let (Some((pa, a)), Some((pb, b))) = (plain(64), plain(4)) else {
            return; // no host GPU here
        };
        let (la, lb) = (64 * PAGE, 4 * PAGE);
        fill(pb, lb, 0x5a);
        let mut batch = Batch::default();
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), la + lb, "all of both at first");
        // Two copies into the big one, a list asking for both after them.
        title_copy(&mut batch, pa + 5 * PAGE, pb, PAGE);
        title_copy(&mut batch, pa + 50 * PAGE, pb, 2 * PAGE);
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(
            batch.flush(),
            3 * PAGE,
            "the bytes written, not the 45 pages between"
        );
        assert_eq!(a.read_host((pa + 5 * PAGE) as u64, 1)[0], 0x5a5a_5a5a);
        assert_eq!(a.read_host((pa + 52 * PAGE - 4) as u64, 1)[0], 0x5a5a_5a5a);
        assert_eq!(a.read_host((pa + 52 * PAGE) as u64, 1)[0], 0, "not past it");
        assert_eq!(
            a.read_host((pa + 5 * PAGE - 4) as u64, 1)[0],
            0,
            "nor before it"
        );
        // And nothing more until the next write.
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), 0);
        // A step that may write anywhere makes everything due again.
        batch.title_written();
        batch.add(&a, pa, la);
        batch.add(&b, pb, lb);
        assert_eq!(batch.flush(), la + lb);
        done(pa, Some(a));
        done(pb, Some(b));
    }

    /// What a title copy wrote is checked again where it is, and only
    /// there, down to the bytes where nothing is tracked.
    #[test]
    fn a_title_copy_within_a_page_is_refreshed_exactly_where_nothing_is_tracked() {
        let Some((p, copy)) = plain(2) else {
            return; // no host GPU here
        };
        fill(p + PAGE, 8, 0x77);
        let mut batch = Batch::default();
        batch.add(&copy, p, 2 * PAGE);
        batch.flush();
        title_copy(&mut batch, p + 100, p + PAGE, 8);
        batch.add(&copy, p, 2 * PAGE);
        let tracking = crate::watch::fault_tracking();
        // With tracking the pages the write marks changed, whole.
        assert_eq!(batch.flush(), if tracking { PAGE } else { 8 });
        assert_eq!(copy.read_host(p as u64 + 100, 1)[0], 0x7777_7777);
        done(p, Some(copy));
    }

    /// A range asked for in part first is current only where it covers
    /// whole pages with tracking (a page taken in part stays changed, and
    /// copy-back reads that as the title's write): the request for all of
    /// the copy takes the ends whole.
    #[test]
    fn a_partial_request_leaves_the_ends_to_the_next() {
        let p = crate::watch::tests::pool(10);
        let len = 10 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            done(p, None);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p + 100, 6 * PAGE);
        assert_eq!(batch.flush(), 6 * PAGE, "just the bytes asked for");
        batch.add(&copy, p, len);
        let tracking = crate::watch::fault_tracking();
        let rest = if tracking { 5 * PAGE } else { 4 * PAGE };
        assert_eq!(
            batch.flush(),
            rest,
            "the pages the first took in part, and the rest"
        );
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0);
        // A fresh submission finds nothing changed.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), if tracking { 0 } else { len });
        done(p, Some(copy));
    }

    /// With tracking, a title copy inside one page leaves the page taken
    /// whole by the request after it, not changed for the next submission.
    #[test]
    fn a_title_copy_inside_a_page_takes_the_page_whole_with_tracking() {
        if !crate::watch::fault_tracking() {
            return;
        }
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            done(p, None);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len);
        let from = [0x33u8; 50];
        title_copy(&mut batch, p + PAGE + 100, from.as_ptr() as usize, 50);
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "the page the copy touched, once");
        assert_eq!(copy.read_host((p + PAGE + 100) as u64, 1)[0], 0x3333_3333);
        let mut next = Batch::default();
        next.add(&copy, p, len);
        assert_eq!(next.flush(), 0, "not left changed for the next submission");
        done(p, Some(copy));
    }

    /// Pages the runtime maps at two addresses: a title copy at one
    /// changes what a copy of the other holds, so it makes everything due
    /// again, as any step that may write anywhere does.
    #[test]
    fn a_title_copy_into_pages_shown_elsewhere_makes_everything_due_again() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let len = 2 * PAGE;
        crate::shadow::register(a, a as u64, len as u64);
        crate::shadow::register(b, b as u64, len as u64);
        let (Some(ca), Some(cb)) = (
            crate::shadow::find(a as u64, len as u64),
            crate::shadow::find(b as u64, len as u64),
        ) else {
            crate::shadow::unregister(a, a as u64);
            crate::shadow::unregister(b, b as u64);
            return; // no host GPU here
        };
        let src = crate::watch::tests::pool(1);
        fill(src, PAGE, 0x42);
        let mut batch = Batch::default();
        batch.add(&ca, a, len);
        batch.add(&cb, b, len);
        batch.flush();
        // Through the first view; the second shows it too.
        title_copy(&mut batch, a, src, PAGE);
        batch.add(&ca, a, len);
        batch.add(&cb, b, len);
        batch.flush();
        assert_eq!(
            cb.read_host(b as u64, 1)[0],
            0x4242_4242,
            "the other view's copy has it"
        );
        assert_eq!(ca.read_host(a as u64, 1)[0], 0x4242_4242);
        crate::shadow::unregister(a, a as u64);
        crate::shadow::unregister(b, b as u64);
        done(src, None);
    }

    #[test]
    fn batch_pool_recycles_cleanly() {
        let mut batch = Batch::take();
        batch.pending.insert(0x1234_5678, 0);
        batch.extents.flushed(0x1234_5678, 0x1000, 0x2000);
        let cap = batch.pending.capacity();
        drop(batch);

        let recycled = Batch::take();
        assert!(recycled.pending.is_empty());
        assert!(recycled.current.is_empty());
        assert!(recycled.refreshes.is_empty());
        assert!(recycled.pending.capacity() >= cap);
    }
}

#[cfg(test)]
mod withholding {
    use crate::backend::HostGpu;

    /// A list in which a program using title memory missed memory: what
    /// that program wrote is withheld from title memory and leaves the
    /// copies at once; what an ordinary program wrote in the same list goes
    /// back and stays in its copy (run light13: the imposter transforms in
    /// the BVH builder's list); an ordinary range overlapping a withheld one
    /// is withheld with it, since its copy-back would take the other's bytes
    /// along.
    #[test]
    fn only_programs_using_title_memory_are_withheld() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let pools = [crate::watch::tests::pool(1), crate::watch::tests::pool(1)];
        for &p in &pools {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(p, p as u64, PAGE as u64);
        }
        let copies: Vec<_> = pools.iter().map(|&p| crate::shadow::find(p as u64, PAGE as u64).unwrap()).collect();
        let list = host.new_list().unwrap();
        // (copy, start, bytes, the writer uses title memory, byte written at +0 and +16)
        let writes = [(0, 0, 32, false, 0x22), (1, 16, 16, true, 0x33), (1, 0, 32, false, 0x44)];
        for &(k, at, len, memory, value) in &writes {
            let _writer = list.writer_uses_memory(memory);
            let (copy, address) = (copies[k].clone(), (pools[k] + at) as u64);
            copy.refresh(&list, address, len);
            copy.gpu_writes(&list, address, len, false);
            // The shader's write, after the step that keeps the range's bytes.
            list.defer_host(move || copy.write_host(address, &[value]));
        }
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let title = |k: usize, at: usize| unsafe { ((pools[k] + at) as *const u8).read_volatile() };
        let copy = |k: usize, at: usize| {
            copies[k].update(pools[k], PAGE);
            copies[k].read_host((pools[k] + at) as u64, 1)[0] & 0xff
        };
        assert_eq!((title(0, 0), copy(0, 0)), (0x22, 0x22), "an ordinary program's write goes back and stays");
        assert_eq!((title(1, 16), copy(1, 16)), (0x11, 0x11), "withheld, then dropped from the copy");
        assert_eq!((title(1, 0), copy(1, 0)), (0, 0), "overlapping a withheld range: withheld with it");
        for (&p, copy) in pools.iter().zip(copies) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }

    /// `Batch` deduplicates repeat `Step::Heap` and `Step::Memory` walks across
    /// lists in a submission while nothing changed, and re-runs them after
    /// `title_written`, a non-aliased `Step::Copy` (`title_copied`), or a
    /// hull/shadow generation change.
    #[test]
    fn batch_deduplicates_heap_and_memory_steps_until_title_write_or_copy() {
        let Some(host) = super::host() else {
            return;
        };
        let p_src = crate::watch::tests::pool(1);
        let p_dst = crate::watch::tests::pool(1);
        let page = crate::watch::PAGE;

        // Unit check on Batch::need_heap / need_memory / title_written / title_copied / Drop pool.
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let mut batch = super::Batch::default();
        assert!(batch.need_heap(&viewed));
        assert!(!batch.need_heap(&viewed), "repeat heap in same batch skipped");
        assert!(batch.need_memory(0x1000));
        assert!(!batch.need_memory(0x1000), "repeat memory heap skipped");
        batch.title_written();
        assert!(batch.need_heap(&viewed), "re-checked after title_written");
        assert!(batch.need_memory(0x1000), "re-checked after title_written");
        assert!(!batch.need_heap(&viewed));
        assert!(!batch.need_memory(0x1000));
        // Non-aliased title_copied must also clear heaps_seen and memories_seen.
        assert!(!crate::watch::aliased(p_dst, 64));
        batch.title_copied(p_dst, 64);
        assert!(batch.need_heap(&viewed), "re-checked after non-aliased title_copied");
        assert!(batch.need_memory(0x1000), "re-checked after non-aliased title_copied");
        // Recycled batch buffers must start with empty seen lists.
        drop(batch);
        let mut recycled = super::Batch::take();
        assert!(recycled.need_heap(&viewed));
        assert!(recycled.need_memory(0x1000));
        drop(recycled);

        // End-to-end check: list 0 has Step::Heap then non-aliased Step::Copy into p_dst;
        // list 1 has Step::Heap and must re-check the heap and pick up the post-Copy bytes.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let buf_desc: [u32; 14] = [1, 0, 0, 0, page as u32, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let res_iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut buf_obj = null;
        assert_eq!(
            crate::resource::create_at_address(
                null,
                p_dst as u64,
                buf_desc.as_ptr().cast(),
                0,
                std::ptr::null(),
                &res_iid,
                &mut buf_obj
            ),
            0
        );
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 16, 0, 1, 0];
        crate::descriptor::create_srv(null, buf_obj, &srv, title_heap.start);
        let heap_host = title_heap.host().expect("host view heap");
        let _ = heap_host.mirror(0, title_heap.start, false);
        let dst_copy = crate::shadow::find(p_dst as u64, page as u64).expect("host copy of p_dst");

        // SAFETY: our pool allocations.
        unsafe {
            (p_src as *mut u32).write_volatile(0xCAFE_BABE);
            (p_dst as *mut u32).write_volatile(0x1111_2222);
        }
        let l0 = host.new_list().unwrap();
        let l1 = host.new_list().unwrap();
        l0.defer_heap(&heap_host.viewed);
        l0.defer_copy(p_dst as *mut u8, p_src as *const u8, 16);
        l1.defer_heap(&heap_host.viewed);
        l0.close();
        l1.close();
        host.execute(&[&l0, &l1]);
        assert_eq!(
            dst_copy.read_host(p_dst as u64, 1)[0],
            0xCAFE_BABE,
            "second list's Step::Heap refreshes bytes written by first list's non-aliased Step::Copy"
        );

        drop(dst_copy);
        crate::object::release_ref(buf_obj);
        crate::object::release_ref(heap);
        crate::watch::tests::free(p_src);
        crate::watch::tests::free(p_dst);
    }
}

#[cfg(test)]
mod heap_events {
    use super::{WidenedEvent, last_heap_events};
    use crate::bindless::Viewed;
    use std::sync::Arc;

    /// A sync expands a heap at its last event only: a later write of the
    /// same heap overrides its earlier ones.
    #[test]
    fn a_heap_is_expanded_at_its_last_event() {
        let (a, b) = (Arc::new(Viewed::default()), Arc::new(Viewed::default()));
        let events = [
            WidenedEvent::Heap(a.clone()),
            WidenedEvent::Heap(b.clone()),
            WidenedEvent::Heap(a.clone()),
        ];
        let last = last_heap_events(&events);
        assert_eq!(last.len(), 2);
        assert_eq!(last.get(&(Arc::as_ptr(&a) as usize)), Some(&2));
        assert_eq!(last.get(&(Arc::as_ptr(&b) as usize)), Some(&1));
    }
}

#[cfg(test)]
mod marks {
    use super::LOST;
    use crate::backend::{HostGpu, HostMemory};
    use std::sync::atomic::Ordering;
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::s;

    /// A submission that ran shows its number where the CPU reads it, so
    /// that one the host GPU drops can be told apart (VIctory-macOS #50).
    #[test]
    fn a_submission_that_ran_shows_its_number() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // Synchronous and without prep-ahead, not while another test runs
        // them: with VICTORY_PREP_AHEAD even submissions are marked by the
        // second mark list, not by `host.mark()`.
        let _mode = super::flight::ModeGuard::new(false, false);
        let mark = host.mark().expect("the mark list");
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        // Under the lock the last submission is the last one marked,
        // whatever other tests submitted meanwhile.
        let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // A flight another test left in the air has marked its number before
        // it ran: land it first.
        host.finish_flight_locked(&mut submitted, None, None);
        assert!(
            mark.saw(*submitted),
            "the mark shows submission {}",
            *submitted
        );
    }

    /// Work goes on, on the same fence, once a new queue replaces the
    /// first, and a queue that runs work passes the check.
    #[test]
    fn work_goes_on_on_a_new_queue() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // Synchronous, and not while another test runs Flight-1: there the
        // submission below returned before it ran, and the check after it
        // failed now and then in the full suite.
        let _mode = super::flight::ModeGuard::new(false, false);
        let mark = host.mark().expect("the mark list");
        let old = host.queue();
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            assert!(host.runs(&mut submitted, mark), "the queue runs work");
            host.replace_queue().unwrap();
            assert!(host.runs(&mut submitted, mark), "the new queue runs work");
        }
        assert!(host.queue() != old, "a new queue");
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        let submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        assert!(mark.saw(*submitted), "a submission after the change ran");
    }

    /// A draw over a `width` x `height` target whose pixels each loop `n`
    /// times, then write `tag` into their own word of a buffer the CPU reads.
    struct Spin {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        _target: ID3D12Resource,
        _rtvs: ID3D12DescriptorHeap,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        buffer: ID3D12Resource,
        words: *const u32,
        width: u32,
        height: u32,
    }

    const SPIN: &str = r#"
RWByteAddressBuffer o : register(u1);
cbuffer c : register(b0) { uint n; uint tag; uint width; }
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps(float4 pos : SV_Position) : SV_Target {
    uint id = (uint)pos.y * width + (uint)pos.x;
    uint a = id;
    for (uint i = 0; i < n; i++) { a = a * 1664525u + 1013904223u; }
    o.Store(id * 8 + 4, a);
    o.Store(id * 8, tag);
    return float4(a & 1, 0, 0, 1);
}
"#;

    impl Spin {
        fn new(host: &super::Host, width: u32, height: u32) -> Spin {
            use crate::compute::Param;
            use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
            let blob = |entry: windows::core::PCSTR, target: windows::core::PCSTR| {
                let mut code: Option<ID3DBlob> = None;
                // SAFETY: the source and names outlive the call.
                unsafe {
                    D3DCompile(
                        SPIN.as_ptr().cast(),
                        SPIN.len(),
                        s!("spin"),
                        None,
                        None,
                        entry,
                        target,
                        0,
                        0,
                        &mut code,
                        None,
                    )
                }
                .unwrap();
                code.unwrap()
            };
            let (vs, ps) = (blob(s!("vs"), s!("vs_5_1")), blob(s!("ps"), s!("ps_5_1")));
            // SAFETY: the blobs' own bytes.
            let bytes = |b: &ID3DBlob| unsafe {
                D3D12_SHADER_BYTECODE {
                    pShaderBytecode: b.GetBufferPointer(),
                    BytecodeLength: b.GetBufferSize(),
                }
            };
            let device = &host.device;
            let root = crate::compute::root_signature(
                device,
                &[
                    Param::Uav { register: 1 },
                    Param::Constants {
                        register: 0,
                        count: 3,
                    },
                ],
            )
            .unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                // SAFETY: borrowed for the call.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                VS: bytes(&vs),
                PS: bytes(&ps),
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            // SAFETY: plain object creation with valid descriptions.
            unsafe {
                let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
                let texture = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(width),
                    Height: height,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let mut target: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &texture,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut target,
                    )
                    .unwrap();
                let target = target.unwrap();
                let rtvs: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
                device.CreateRenderTargetView(&target, None, rtv);
                let words = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(width * height) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let mut buffer: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_CUSTOM,
                            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &words,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                let mut at = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut at)).unwrap();
                Spin {
                    root,
                    pso,
                    _target: target,
                    _rtvs: rtvs,
                    rtv,
                    buffer,
                    words: at.cast_const().cast(),
                    width,
                    height,
                }
            }
        }

        /// Draws; how long the submission took (ms) and how many pixels
        /// wrote `tag`.
        fn draw(&self, host: &super::Host, n: u32, tag: u32) -> (f64, usize) {
            let list = host.new_list().unwrap();
            let l = &list.list;
            // SAFETY: an open list; our pipeline, buffer and target.
            unsafe {
                l.SetPipelineState(&self.pso);
                l.SetGraphicsRootSignature(&self.root);
                l.SetGraphicsRootUnorderedAccessView(0, self.buffer.GetGPUVirtualAddress());
                l.SetGraphicsRoot32BitConstants(1, 3, [n, tag, self.width].as_ptr().cast(), 0);
                l.OMSetRenderTargets(1, Some(&self.rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: self.width as f32,
                    Height: self.height as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: self.width as i32,
                    bottom: self.height as i32,
                }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                l.DrawInstanced(3, 1, 0, 0);
            }
            list.close();
            let started = std::time::Instant::now();
            host.execute(&[&list]);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let pixels = (self.width * self.height) as usize;
            // SAFETY: our mapped buffer of 2 words per pixel.
            let ran = (0..pixels)
                .filter(|&i| unsafe { self.words.add(2 * i).read_volatile() } == tag)
                .count();
            (ms, ran)
        }
    }

    /// Under D3DMetal, macOS ends a draw whose every pixel runs long about
    /// 50 ms in, and after some of those the queue ignores all later work
    /// while its fence still completes (VIctory-macOS #50). The ended
    /// submissions are seen, a new queue replaces the one that stopped, and
    /// a draw after that runs whole. Wine on macOS only, and ignored:
    /// elsewhere these draws would run for seconds or reset the GPU.
    #[test]
    #[ignore]
    fn work_goes_on_after_macos_stops_the_queue() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (width, height) = (512, 256);
        let pixels = (width * height) as usize;
        let spin = Spin::new(host, width, height);
        let mut tag = 1;
        // A first draw, not timed: on an M1 Max (macOS 15) the first draw
        // took 24.5 ms where a warm one of the same size takes 3.7 ms, so
        // the calibration stopped there, came out about 100 times too slow,
        // and the "4 s" draws ran whole in 43 ms.
        spin.draw(host, 1, tag);
        // Loop passes per millisecond, from a draw of 20 to 40 ms.
        let mut n = 1024u32;
        let per_ms = loop {
            tag += 1;
            let (ms, ran) = spin.draw(host, n, tag);
            assert_eq!(ran, pixels, "a short draw runs whole");
            if ms >= 20.0 {
                break f64::from(n) / ms;
            }
            n *= 2;
        };
        let long = (per_ms * 4000.0).min(f64::from(u32::MAX)) as u32;
        let [lost, queues] = [0, 1].map(|i| LOST[i].load(Ordering::Relaxed));
        // On an M1 Max (macOS 15) the queue stopped after 2 to 30 ended
        // draws (12 runs), each about 260 ms, so 8 tries often fell short.
        for _ in 0..64 {
            tag += 1;
            let (_, ran) = spin.draw(host, long, tag);
            assert_ne!(ran, pixels, "macOS ends a draw of about 4 s");
            if LOST[1].load(Ordering::Relaxed) > queues {
                break;
            }
        }
        assert!(
            LOST[0].load(Ordering::Relaxed) > lost,
            "the ended draws were seen"
        );
        assert!(
            LOST[1].load(Ordering::Relaxed) > queues,
            "a new queue replaced the one that stopped"
        );
        tag += 1;
        assert_eq!(
            spin.draw(host, 4, tag).1,
            pixels,
            "a draw on the new queue runs whole"
        );
    }

    #[test]
    fn bury_lag_retains_retired_resources_across_submissions() {
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let references = |b: &ID3D12Resource| {
            // SAFETY: a live COM object; one reference taken and given back.
            unsafe {
                let raw = b.as_raw();
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            }
        };
        let wait_released = |b: &ID3D12Resource| {
            let since = std::time::Instant::now();
            while references(b) > 1 && since.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
            references(b)
        };

        let (buf_lag, _) = host.shared_buffer(256).unwrap();
        let list = host.new_list().unwrap();
        list.close();
        super::with_bury_lag(3, || {
            let _quiet = host.quiet();
            // The submission count is global: tests that run at the same
            // time add to it, so retention is checked against the count the
            // retirement could not have been recorded after.
            let before = super::submissions_done();
            super::retire(buf_lag.clone());
            assert!(references(&buf_lag) >= 2, "held by retirement ring");
            drop(_quiet);

            for step in 1..3 {
                list.reset();
                list.close();
                host.execute(&[&list]);
                let passed = super::submissions_done().saturating_sub(before);
                assert!(
                    passed >= 3 || references(&buf_lag) >= 2,
                    "still retained after {step} of 3 lag submissions ({passed} passed)"
                );
            }
            list.reset();
            list.close();
            host.execute(&[&list]);
            assert_eq!(wait_released(&buf_lag), 1, "released once 3 submissions passed");
        });

        let (buf_immediate, _) = host.shared_buffer(256).unwrap();
        super::with_bury_lag(0, || {
            super::retire(buf_immediate.clone());
            list.reset();
            list.close();
            host.execute(&[&list]);
            assert_eq!(
                wait_released(&buf_immediate),
                1,
                "VICTORY_BURY_LAG=0 releases right after the submission"
            );
        });
    }

    /// Background heap-worker staging runs concurrently with `Host::quiet()` /
    /// `HostHeap::new()` and `Host::execute()` without deadlock or lock
    /// inversion (verifies Issue #31 compatibility).
    #[test]
    fn worker_concurrent_with_quiet_heap_creation_does_not_deadlock() {
        let Some(host) = super::host() else {
            return;
        };
        let _worker = crate::heap_worker::EnableForTest::new();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 32, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let _ = title_heap.host().expect("host view heap");
        let pool = crate::watch::tests::pool(1);
        // SAFETY: dedicated pool page.
        unsafe { std::ptr::write_bytes(pool as *mut u8, 0x22, crate::watch::PAGE) };
        let placed_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut desc = [0u32; 16];
        (desc[0], desc[1], desc[2], desc[5]) = (28, 4, 0x1688, 1);
        (desc[10], desc[11]) = (pool as u32, (pool >> 32) as u32);
        for i in 0..8u64 {
            crate::descriptor::create_placed_srv(null, &placed_res, &desc, title_heap.start + 32 * i);
        }
        assert!(crate::heap_worker::kick(title_heap));
        let quiet_thread = std::thread::spawn(move || {
            for _ in 0..4 {
                let _q = host.quiet();
                let _ = crate::descriptor::host::HostHeap::new(false, 64);
            }
        });
        let list = host.new_list().unwrap();
        for _ in 0..4 {
            list.close();
            host.execute(&[&list]);
            list.reset();
        }
        quiet_thread.join().unwrap();
        crate::object::release_ref(heap);
        crate::watch::tests::free(pool);
    }
}

/// The pipelines a list binds ([`HostList::pipelines`]).
type Pipelines = std::sync::Arc<std::collections::BTreeSet<String>>;

/// The lists the last submission ran (how many, and their pipelines), for a
/// hang's log line (`Host::wait_for`): which work the GPU never finished.
/// Naming them is left to that line: building the names for every
/// submission cost 0.2 ms each under Wine at 100 pipelines a list, 0.6 ms
/// at 300 (benchmark `waiting_on_takes_the_sets`).
static WAITING_ON: Mutex<(usize, Vec<Pipelines>)> = Mutex::new((0, Vec::new()));

/// Empties a list's pipelines at its reset, leaving a set the hang note
/// still holds as it is.
fn reset_pipelines(pipelines: &mut Pipelines) {
    match std::sync::Arc::get_mut(pipelines) {
        Some(set) => set.clear(),
        None => *pipelines = Pipelines::default(),
    }
}

/// Every pipeline name of `sets`, once each and in order.
fn pipeline_names(sets: impl Iterator<Item = Pipelines>) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    for set in sets {
        names.extend(set.iter().cloned());
    }
    names
}

#[cfg(test)]
mod waiting_on_tests {
    use super::{Pipelines, pipeline_names, reset_pipelines};
    use std::sync::Arc;

    fn sets(lists: usize, per: usize) -> Vec<Pipelines> {
        (0..lists)
            .map(|l| {
                Arc::new(
                    (0..per)
                        .map(|p| format!("vs_{:016x}_ps_{:016x}", p * 7 + l, p))
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn a_reset_leaves_the_hang_notes_set_whole() {
        let mut list: Pipelines = Arc::new(["a".to_owned(), "b".to_owned()].into());
        let noted = list.clone();
        reset_pipelines(&mut list);
        assert!(list.is_empty());
        assert_eq!(noted.len(), 2, "the note keeps what the submission ran");
        // Unshared, the list's own set is emptied in place.
        let before = Arc::as_ptr(&list);
        Arc::make_mut(&mut list).insert("c".to_owned());
        reset_pipelines(&mut list);
        assert!(list.is_empty());
        assert_eq!(Arc::as_ptr(&list), before);
        assert_eq!(
            pipeline_names([noted, Arc::new(["b".to_owned(), "z".to_owned()].into())].into_iter())
                .into_iter()
                .collect::<Vec<_>>(),
            ["a", "b", "z"]
        );
    }

    /// What noting a submission costs: copying every name (as before) against
    /// taking the sets. `cargo test --release -p victory-gpu --lib
    /// waiting_on_takes_the_sets -- --ignored --nocapture`
    #[test]
    #[ignore = "times the hang note of a submission"]
    fn waiting_on_takes_the_sets() {
        for per in [20, 100, 300] {
            let lists = sets(10, per);
            let rounds = 2000;
            let started = std::time::Instant::now();
            let mut total = 0;
            for _ in 0..rounds {
                let names = pipeline_names(lists.iter().cloned());
                total += format!("{} lists, pipelines {names:?}", lists.len()).len();
            }
            let copied = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
            let mut held: Vec<Pipelines> = Vec::new();
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                held.clear();
                held.extend(lists.iter().cloned());
                total += held.len();
            }
            let taken = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
            println!("{per} pipelines a list: names {copied:.1} us, sets {taken:.2} us ({total})");
        }
    }
}

/// `VICTORY_PHYSICAL_TILES=1`: a title buffer on physical pages that are GPU
/// memory is read in place (`crate::tiled`).
#[cfg(test)]
mod tiled_tests {
    use crate::alias::{PhysicalHeap, Tile};
    use crate::backend::{HostGpu, HostMemory};
    use crate::watch::PAGE;
    use std::sync::Mutex;

    /// What the test's "runtime" maps: page address and its tile.
    static SHOWN: Mutex<Vec<(usize, Tile)>> = Mutex::new(Vec::new());

    fn lookup(at: usize, len: usize, out: &mut Vec<Option<Tile>>) -> bool {
        let shown = SHOWN.lock().unwrap();
        out.clear();
        out.extend((at / PAGE..(at + len).div_ceil(PAGE)).map(|page| {
            shown.iter().find(|(a, _)| *a == page * PAGE).map(|(_, t)| t.clone())
        }));
        true
    }

    /// Title memory of four pages showing heap tiles 2, 0, 3, 1 in turn.
    #[test]
    fn a_buffer_on_gpu_physical_pages_is_read_and_written_in_place() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _mode = super::flight::ModeGuard::new(false, false);
        crate::alias::TEST_PHYSICAL_TILES.with(|c| c.set(Some(true)));
        if !crate::alias::physical_tiles() {
            crate::alias::TEST_PHYSICAL_TILES.with(|c| c.set(None));
            return; // not Rosetta and D3DMetal
        }
        crate::tiled::on_physical_tiles(lookup);
        let heap = PhysicalHeap::new(&host.device, 4 * PAGE).expect("a CPU-visible heap");
        // SAFETY: a fresh reservation of four pages, never freed (its
        // addresses stay the test's).
        let at = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                windows_sys::Win32::System::Memory::MEM_RESERVE | windows_sys::Win32::System::Memory::MEM_COMMIT,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(at, 0);
        let order = [2usize, 0, 3, 1];
        for (k, &tile) in order.iter().enumerate() {
            // SAFETY: our reservation's page and the heap's tile.
            assert!(unsafe { heap.alias_view(at + k * PAGE, tile * PAGE, PAGE) }, "remap page {k}");
            SHOWN.lock().unwrap().push((at + k * PAGE, heap.tile(tile * PAGE).unwrap()));
        }
        // The title writes each page through its own address.
        for k in 0..4 {
            // SAFETY: mapped read-write.
            unsafe { ((at + k * PAGE) as *mut u32).write_volatile(0x1000 + k as u32) };
        }
        let resource = 0x7e57_0000_usize;
        crate::shadow::register(resource, at as u64, (4 * PAGE) as u64);
        let copy = crate::shadow::find(at as u64, (4 * PAGE) as u64).expect("a copy");
        assert!(copy.is_aliased(), "the copy reads the pages in place");

        // The GPU reads what the title wrote, with no refresh.
        let (out, out_ptr) = host.mapped_buffer(64, true).expect("readback");
        let read = |page: usize| {
            // SAFETY: mapped readback memory, no copy into it in flight.
            unsafe { out_ptr.cast::<u32>().write_volatile(0xDEAD_DEAD) };
            let list = host.new_list().unwrap();
            // SAFETY: open list, live buffers.
            unsafe { list.list.CopyBufferRegion(&out, 0, &copy.buffer, copy.offset((at + page * PAGE) as u64), 4) };
            list.keep(&out);
            list.close();
            host.execute(&[&list]);
            // SAFETY: the copy has run.
            unsafe { out_ptr.cast::<u32>().read_volatile() }
        };
        for k in 0..4 {
            assert_eq!(read(k), 0x1000 + k as u32, "page {k} as the GPU reads it");
        }
        // What the GPU writes the title reads at once.
        let (src, src_ptr) = host.upload_buffer(64).expect("upload");
        // SAFETY: mapped upload memory.
        unsafe { src_ptr.cast::<u32>().write_volatile(0xBEEF) };
        let list = host.new_list().unwrap();
        // SAFETY: open list, live buffers.
        unsafe { list.list.CopyBufferRegion(&copy.buffer, copy.offset((at + 2 * PAGE + 8) as u64), &src, 0, 4) };
        list.keep(&src);
        list.close();
        host.execute(&[&list]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { ((at + 2 * PAGE + 8) as *const u32).read_volatile() }, 0xBEEF);

        // The title maps tile 3 at its second page as well: the copy follows.
        // SAFETY: as above.
        assert!(unsafe { heap.alias_view(at + PAGE, 3 * PAGE, PAGE) });
        crate::tiled::physical_retiled(at + PAGE, PAGE, &[heap.tile(3 * PAGE)]);
        assert_eq!(read(1), 0x1002, "the second page shows tile 3 now");
        // Unmapped, it reads zeros (D3DMetal ignores null tile mappings: a
        // tile of zeros stands in), and the other pages are as they were.
        crate::tiled::physical_retiled(at + PAGE, PAGE, &[None]);
        assert_eq!(read(1), 0, "an unmapped page reads zeros");
        assert_eq!([read(0), read(3)], [0x1000, 0x1003]);

        crate::shadow::unregister(resource, at as u64);
        SHOWN.lock().unwrap().clear();
        crate::alias::TEST_PHYSICAL_TILES.with(|c| c.set(None));
    }

    /// A buffer with a page that is not GPU memory is copied as before.
    #[test]
    fn a_buffer_off_gpu_pages_is_copied() {
        if super::host().is_none() {
            return;
        }
        crate::alias::TEST_PHYSICAL_TILES.with(|c| c.set(Some(true)));
        crate::tiled::on_physical_tiles(lookup);
        let p = crate::watch::tests::pool(2);
        crate::shadow::register(p, p as u64, (2 * PAGE) as u64);
        let copy = crate::shadow::find(p as u64, (2 * PAGE) as u64).expect("a copy");
        assert!(!copy.is_aliased());
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
        crate::alias::TEST_PHYSICAL_TILES.with(|c| c.set(None));
    }
}
