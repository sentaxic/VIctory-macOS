//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex or mesh plus pixel pipelines from streams and
//! compute pipelines get host pipelines; amplification, tessellation and
//! geometry stages, and the plain graphics description, keep their
//! containers only. The 32-byte packet after the header stays zero: titles
//! only copy it inline, which our null command-list limits keep from
//! running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
#[cfg(windows)]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
#[cfg(windows)]
use victory_shader::emit::{metal_mesh_output_bytes, over_limit_mesh_output_bytes};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    #[cfg(windows)]
    full_hash: [u8; 32],
    /// The container's length, as logged once its bytes are gone.
    size: usize,
    /// Released after translation unless an experimental mesh/pixel pair may need it.
    #[cfg(any(windows, test))]
    container: Mutex<Vec<u8>>,
    /// Its full BLAKE3 once the container is on disk instead
    /// (`VICTORY_SPILL_SHADERS`, [`spill`]).
    #[cfg(windows)]
    spilled: OnceLock<[u8; 32]>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut created = false;
    let shader = {
        let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
        shaders
            .get_or_insert_with(HashMap::new)
            .entry(*full.as_bytes())
            .or_insert_with(|| {
                created = true;
                // The programs the ambient stand-in follows are known by name
                // (`crate::ambient`; nothing while it is off).
                #[cfg(windows)]
                crate::ambient::note_program(&full.to_hex()[..16], bytes);
                // So are the ray passes `VICTORY_RAY_SKIP` leaves out
                // (`crate::ray_skip`; nothing while it is off).
                #[cfg(windows)]
                crate::ray_skip::note_program(stage, &full.to_hex()[..16], bytes);
                Arc::new(Shader::with_hash(stage, bytes, full))
            })
            .clone()
    };
    // Written outside the lock every pipeline creation takes.
    #[cfg(windows)]
    if created {
        spill(&shader, full.as_bytes());
        // With the frame recorder asked for (`VICTORY_RECORD`), the container
        // is kept beside its captures, so that a replay can translate it
        // again (`--retranslate`); without it, one look at a flag.
        crate::replay::record::container(&full.to_hex()[..16], bytes);
    }
    #[cfg(not(windows))]
    let _ = created;
    shader
}

/// Folder for shader containers kept on disk until translated
/// (`VICTORY_SPILL_SHADERS=1`): the title passes tens of thousands of
/// distinct containers and draws with a few hundred, and holding them all
/// was ~700 MB of an 8 GB Mac's memory (#135).
#[cfg(windows)]
static SPILL_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The file a container with BLAKE3 `full` is kept in.
#[cfg(windows)]
fn spill_path(dir: &std::path::Path, full: &[u8; 32]) -> std::path::PathBuf {
    let hex = blake3::Hash::from(*full).to_hex();
    dir.join(&hex[..2]).join(format!("{hex}.bin"))
}

/// Moves `shader`'s container to disk if spilling is on: written under a
/// temporary name and renamed, so a file found there is whole; one already
/// there from an earlier run is kept.
#[cfg(windows)]
fn spill(shader: &Shader, full: &[u8; 32]) {
    if let Some(dir) = SPILL_DIR.get() {
        spill_to(dir, shader, full);
    }
}

/// [`spill`] into `dir`.
#[cfg(windows)]
fn spill_to(dir: &std::path::Path, shader: &Shader, full: &[u8; 32]) {
    let path = spill_path(dir, full);
    let there = std::fs::File::open(&path).is_ok();
    if !there {
        let Some(shard) = path.parent() else { return };
        let mut container = shader.container.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let ok = crate::make_dir(shard).is_ok()
            && std::fs::write(&tmp, &*container).is_ok()
            && std::fs::rename(&tmp, &path).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        *container = Vec::new();
    } else {
        *shader.container.lock().unwrap_or_else(|e| e.into_inner()) = Vec::new();
    }
    let _ = shader.spilled.set(*full);
}

impl Shader {
    fn new(stage: &'static str, bytes: &[u8]) -> Shader {
        Self::with_hash(stage, bytes, blake3::hash(bytes))
    }

    fn with_hash(stage: &'static str, bytes: &[u8], digest: blake3::Hash) -> Shader {
        Shader {
            stage,
            hash: digest.to_hex()[..16].to_owned(),
            #[cfg(windows)]
            full_hash: *digest.as_bytes(),
            size: bytes.len(),
            #[cfg(any(windows, test))]
            container: Mutex::new(bytes.to_vec()),
            #[cfg(windows)]
            spilled: OnceLock::new(),
            #[cfg(windows)]
            translated: OnceLock::new(),
        }
    }

    #[cfg(all(test, windows))]
    pub(crate) fn for_test(stage: &'static str, translated: Translated) -> Arc<Shader> {
        let shader = Shader::new(stage, b"");
        let _ = shader.translated.set(Some(translated));
        Arc::new(shader)
    }
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Per [`Translated::used`] entry: a buffer store or atomic names that
    /// table UAV descriptor. Empty when the shader's code is not known.
    pub(crate) used_written: Vec<bool>,
    /// Whether any buffer store or atomic could not be matched to a known
    /// descriptor in [`Translated::used`], so all table UAVs keep writeback.
    pub(crate) write_unknown: bool,
    /// A compute program's threads in a group (`numthreads`), which a
    /// dispatch counted in threads divides by (`draw::dispatch_groups`).
    pub(crate) threads: Option<[u32; 3]>,
}

#[cfg(windows)]
impl Shader {
    /// Its full BLAKE3, kept whether or not the container's bytes still are.
    pub(crate) fn full_hash(&self) -> [u8; 32] {
        self.full_hash
    }

    /// The translation, made on first use (or taken from the cache); a
    /// shader that does not translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let (started, cpu) = (std::time::Instant::now(), thread_cpu());
                let (done, cached) = {
                    let mut container = self.container.lock().unwrap_or_else(|e| e.into_inner());
                    if container.is_empty()
                        && let (Some(full), Some(dir)) = (self.spilled.get(), SPILL_DIR.get())
                    {
                        *container = crate::read_file(&spill_path(dir, full)).unwrap_or_default();
                    }
                    let done = translate(&container);
                    // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title
                    // gave it (`.cdxil`) and as translated (`.dxil`), by the
                    // hash the log names it with; `victory translate` with
                    // `VICTORY_TRANSLATE_ONLY` takes the same hash.
                    if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                        let dir = std::path::PathBuf::from(dir);
                        let _ = std::fs::create_dir_all(&dir);
                        let stem = format!("{}.{}", self.hash, self.stage);
                        let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &container[..]);
                        if let Ok(t) = &done.0 {
                            let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &t.host.dxil);
                        }
                    }
                    // After the translation, not before: if it panics, the
                    // bytes are still there for the next try.
                    let keep = (link_mesh_outputs()
                        && done.0.as_ref().is_ok_and(|t| {
                            self.stage == "pixel"
                                || (self.stage == "mesh"
                                    && over_limit_mesh_output_bytes(&t.host.dxil).is_some())
                        }))
                        || (mesh_per_vertex_center()
                            && self.stage == "pixel"
                            && done.0.as_ref().is_ok_and(|t| {
                                victory_shader::emit::needs_vertex_deltas(&t.host.dxil)
                            }));
                    // A compute program that may leave the lanes past a
                    // dispatch's count of threads is translated again for
                    // that if one comes (`thread_limited`).
                    let limits = done.0.as_ref().is_ok_and(|t| may_limit_threads(t.threads));
                    if !keep && !limits {
                        *container = Vec::new();
                    }
                    done
                };
                let ms = started.elapsed().as_millis() as u64;
                let (user_ms, kernel_ms) = thread_cpu_since(cpu);
                match &done {
                    Ok(t) => {
                        tracing::info!(
                            stage = self.stage,
                            hash = %self.hash,
                            user_data = t.host.user_data.len(),
                            samplers = t.host.samplers.len(),
                            cached,
                            ms,
                            user_ms,
                            kernel_ms,
                            "shader translated"
                        );
                        // D3DMetal makes such a pipeline and marks it a no-op
                        // or hangs the queue (#132, #172).
                        use victory_shader::emit::{METAL_MESH_OUTPUT_BYTES, mesh_outputs};
                        if let (Some(o), Some(bytes)) =
                            (mesh_outputs(&t.host.dxil), metal_mesh_output_bytes(&t.host.dxil))
                            && bytes > METAL_MESH_OUTPUT_BYTES
                            && crate::shader_cache::under_wine()
                        {
                            tracing::warn!(
                                hash = %self.hash,
                                bytes,
                                packed_bytes = o.bytes(),
                                limit = METAL_MESH_OUTPUT_BYTES,
                                vertices = o.vertices,
                                primitives = o.primitives,
                                vertex_components = o.vertex_components,
                                primitive_components = o.primitive_components,
                                "mesh shader outputs exceed Metal's limit; linking or refusing on host"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        cached,
                        ms,
                        user_ms,
                        kernel_ms,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

/// This thread's processor time so far, user and kernel, in 100 ns units.
#[cfg(windows)]
fn thread_cpu() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let mut t = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [created, exited, kernel, user] = &mut t;
    // SAFETY: four times of our own thread into our own variables.
    unsafe { GetThreadTimes(GetCurrentThread(), created, exited, kernel, user) };
    let v = |f: &FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    (v(user), v(kernel))
}

/// This thread's processor time since `then` ([`thread_cpu`]): user and
/// kernel milliseconds. Against the wall time, it tells work from waiting
/// (for a processor, a lock, the disk).
#[cfg(windows)]
fn thread_cpu_since(then: (u64, u64)) -> (u64, u64) {
    let now = thread_cpu();
    ((now.0 - then.0) / 10_000, (now.1 - then.1) / 10_000)
}

/// The translation, and whether it came from the cache.
#[cfg(windows)]
fn translate(container: &[u8]) -> (Result<Translated, String>, bool) {
    let (host, cached) = crate::shader_cache::host_shader(container);
    (host.and_then(|host| with_resources(container, host)), cached)
}

#[cfg(windows)]
fn with_resources(
    container: &[u8],
    host: victory_shader::emit::HostShader,
) -> Result<Translated, String> {
    use victory_shader::native::StageIo;
    use victory_shader::{Container, FourCC, HwStage};
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let chosen = programs.iter().find(|p| {
        matches!(
            p.stage,
            HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
        )
    });
    let threads = chosen.and_then(|p| match p.io {
        StageIo::Cs(cs) if cs.threads.iter().all(|&n| n > 0) => Some(cs.threads),
        _ => None,
    });
    let used = chosen.map(|p| p.used.clone()).unwrap_or_default();
    let (used_written, mut write_unknown) = chosen
        .map(|p| written_used(p, &used))
        .unwrap_or((Vec::new(), !programs.is_empty()));
    if programs.len() > 1 {
        // Another stage in the container could store through a shared table.
        write_unknown = true;
    }
    Ok(Translated {
        host,
        used,
        used_written,
        write_unknown,
        threads,
    })
}

/// Which [`Translated::used`] table UAV descriptors a buffer store or atomic
/// names. Traces every buffer/SMEM store or atomic through
/// [`victory_shader::resource::track`] back to its root-table `(user_data,
/// table_offset_dwords)` descriptor; any store whose target cannot be resolved
/// to a known root or table descriptor (or goes through the dynamic resource
/// heap) sets `write_unknown` so all table UAVs keep writeback.
#[cfg(windows)]
fn written_used(
    program: &victory_shader::NativeProgram,
    used: &[victory_shader::UsedResource],
) -> (Vec<bool>, bool) {
    use std::collections::HashMap;
    use victory_shader::gfx10::Encoding;
    use victory_shader::resource::{Role, Source};
    use victory_shader::{ResourceType, RootType};

    let Ok(ir) = victory_shader::ir::build(&program.code, program.wave64, program.vgprs) else {
        return (vec![true; used.len()], true);
    };
    let st = victory_shader::structure::structure(&ir).ok();
    let Ok(res) = victory_shader::resource::track(&ir, &program.layout, program.stage, st.as_ref())
    else {
        return (vec![true; used.len()], true);
    };
    let scalar: HashMap<_, _> = res.scalar.iter().copied().collect();
    let descriptors: HashMap<_, _> = res
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource)
        .map(|d| (d.site, &d.source))
        .collect();

    let mut written = vec![false; used.len()];
    let mut unknown = false;
    for (bi, block) in ir.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            let is_buffer_write = match inst.machine.encoding {
                Encoding::Mubuf | Encoding::Mtbuf | Encoding::Smem => {
                    let name = inst.name();
                    name.is_empty() || name.contains("store") || name.contains("atomic")
                }
                _ => false,
            };
            if !is_buffer_write {
                continue;
            }
            let Some(source) = descriptors.get(&(bi, ii)) else {
                // `s_store_*` / `s_atomic_*` write via a scalar address rather
                // than a descriptor; root descriptors are tracked separately.
                if let Some(addr) = scalar.get(&(bi, ii))
                    && program
                        .layout
                        .get(addr.param)
                        .is_some_and(|rp| rp.root == RootType::Descriptor)
                {
                    continue;
                }
                unknown = true;
                continue;
            };
            for leaf in source.leaves() {
                match leaf {
                    Source::Loaded { load, first } => {
                        let Some(addr) = scalar.get(load) else {
                            unknown = true;
                            continue;
                        };
                        let Some(rp) = program.layout.get(addr.param) else {
                            unknown = true;
                            continue;
                        };
                        match rp.root {
                            RootType::Table => {
                                if addr.dynamic.is_some() {
                                    let mut any = false;
                                    for (i, u) in used.iter().enumerate() {
                                        if u.root == RootType::Table
                                            && u.user_data == rp.user_data
                                            && u.kind == ResourceType::Uav
                                        {
                                            written[i] = true;
                                            any = true;
                                        }
                                    }
                                    if !any {
                                        unknown = true;
                                    }
                                } else {
                                    let dword = (addr.offset / 4) + u32::from(*first);
                                    let hit = used
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, u)| {
                                            u.root == RootType::Table
                                                && u.user_data == rp.user_data
                                                && u.table_offset_dwords <= dword
                                        })
                                        .max_by_key(|(_, u)| u.table_offset_dwords);
                                    if let Some((i, u)) = hit
                                        && u.kind == ResourceType::Uav
                                    {
                                        written[i] = true;
                                    } else {
                                        unknown = true;
                                    }
                                }
                            }
                            RootType::Descriptor => {}
                            _ => {
                                unknown = true;
                            }
                        }
                    }
                    Source::Built { address, .. }
                        if program
                            .layout
                            .get(address.param)
                            .is_some_and(|rp| rp.root == RootType::Descriptor) => {}
                    _ => {
                        unknown = true;
                    }
                }
            }
        }
    }
    (written, unknown)
}

/// Whether a compute program of `threads` per group could leave the lanes
/// past a dispatch's count of threads ([`thread_limited`]): one dimension,
/// more than one thread. Its container is kept for that translation.
#[cfg(windows)]
fn may_limit_threads(threads: Option<[u32; 3]>) -> bool {
    threads.is_some_and(|t| t[0] > 1 && t[1..] == [1, 1])
}

/// The host pipeline of compute pipeline `p` whose lanes past a dispatch's
/// count of threads leave at once (`emit::Options::thread_limits`), made on
/// the first dispatch counted in threads that ends in a part of a group
/// (`draw::State::dispatch_x`): its shader translated again, so programs
/// no such dispatch reaches keep their translation. None where the program
/// cannot leave them (a group barrier, 32-lane halves passing values) or
/// that translation or pipeline fails; asked once per shader.
#[cfg(windows)]
pub(crate) fn thread_limited(p: &host::HostPipeline) -> Option<&'static host::HostPipeline> {
    type Made = HashMap<[u8; 32], Option<&'static host::HostPipeline>>;
    static MADE: Mutex<Option<Made>> = Mutex::new(None);
    let shader = &p.stages.first().filter(|_| p.compute)?.shader;
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&hit) = made.as_ref().and_then(|m| m.get(&shader.full_hash)) {
        return hit;
    }
    let started = std::time::Instant::now();
    let limited = make_thread_limited(shader);
    tracing::info!(
        hash = %shader.hash,
        made = limited.is_some(),
        ms = started.elapsed().as_millis() as u64,
        "a compute pipeline whose lanes past a dispatch's count of threads leave"
    );
    made.get_or_insert_with(HashMap::new)
        .insert(shader.full_hash, limited);
    limited
}

#[cfg(windows)]
fn make_thread_limited(shader: &Shader) -> Option<&'static host::HostPipeline> {
    let t = shader.translated()?;
    let mut bytes = shader
        .container
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if bytes.is_empty()
        && let (Some(full), Some(dir)) = (shader.spilled.get(), SPILL_DIR.get())
    {
        bytes = crate::read_file(&spill_path(dir, full)).unwrap_or_default();
    }
    if bytes.is_empty() {
        tracing::warn!(hash = %shader.hash, "no container left to translate with thread limits");
        return None;
    }
    let options = victory_shader::emit::Options {
        thread_limits: true,
        ..crate::shader_cache::runtime_options()
    };
    let host = match crate::shader_cache::host_shader_with(&bytes, options).0 {
        Ok(host) => host,
        Err(e) => {
            tracing::warn!(hash = %shader.hash, error = %e, "no translation with thread limits");
            return None;
        }
    };
    let kind = victory_shader::RootType::Other(victory_shader::emit::USER_DISPATCH_THREADS_X);
    if !host.user_data.iter().any(|u| u.root == kind) {
        return None;
    }
    let limited = shader.with_translation(Translated {
        host,
        used: t.used.clone(),
        used_written: t.used_written.clone(),
        write_unknown: t.write_unknown,
        threads: t.threads,
    });
    let p = host::HostPipeline::new(&[limited], None, |_, _| {})?;
    Some(Box::leak(Box::new(p)))
}

#[cfg(windows)]
fn link_mesh_outputs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // On by default under Wine (D3DMetal): an over-limit mesh pipeline there
    // is made, draws nothing and its submission never completes (#172).
    *ON.get_or_init(|| match std::env::var_os("VICTORY_LINK_MESH_OUTPUTS") {
        Some(v) => crate::switch::value_is_on(Some(&v)),
        None => crate::shader_cache::under_wine(),
    })
}

/// Whether mesh+pixel pipelines whose pixel shader reads non-provoking vertex
/// values (`VICTORY_P10`/`VICTORY_P20`) fall back to center interpolation
/// (`per_vertex_center`) instead of being refused (`VICTORY_MESH_PER_VERTEX_CENTER=0`
/// disables). Default on.
#[cfg(windows)]
pub(crate) fn mesh_per_vertex_center() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mesh_per_vertex_center_value(std::env::var_os("VICTORY_MESH_PER_VERTEX_CENTER").as_deref())
    })
}

#[cfg(any(windows, test))]
fn mesh_per_vertex_center_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

/// Whether a mesh pipeline whose outputs are still over Metal's 32 KiB after
/// linking is refused rather than made. On by default under Wine: on
/// D3DMetal (3.0 and 4.0 beta 3) such a pipeline hangs the queue for good,
/// and the title's watchdog then ends the run (#172). Its draws are
/// dropped instead. `VICTORY_MESH_OVER_LIMIT=1` makes it anyway.
#[cfg(windows)]
pub(crate) fn refuse_over_limit_mesh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_MESH_OVER_LIMIT") {
        Some(v) => !crate::switch::value_is_on(Some(&v)),
        None => crate::shader_cache::under_wine(),
    })
}

#[cfg(windows)]
struct LinkedPair {
    mesh: Arc<Shader>,
    pixel: Option<Arc<Shader>>,
}

#[cfg(windows)]
type LinkedKey = ([u8; 32], Option<[u8; 32]>, bool);

#[cfg(windows)]
impl Shader {
    fn container_bytes(&self) -> Vec<u8> {
        let container = self.container.lock().unwrap_or_else(|e| e.into_inner());
        if !container.is_empty() {
            return container.clone();
        }
        drop(container);
        if let (Some(full), Some(dir)) = (self.spilled.get(), SPILL_DIR.get()) {
            return crate::read_file(&spill_path(dir, full)).unwrap_or_default();
        }
        Vec::new()
    }

    fn with_translation(&self, translated: Translated) -> Arc<Shader> {
        let cell = OnceLock::new();
        let _ = cell.set(Some(translated));
        Arc::new(Shader {
            stage: self.stage,
            hash: self.hash.clone(),
            full_hash: self.full_hash,
            size: self.size,
            container: Mutex::new(Vec::new()),
            spilled: OnceLock::new(),
            translated: cell,
        })
    }
}

/// Re-translates a mesh shader together with its paired pixel shader (if any)
/// so arbitrary mesh outputs unread by the pixel shader are pruned under
/// Metal's 32 KiB mesh output limit (`victory_shader::emit::METAL_MESH_OUTPUT_BYTES`).
/// When `per_vertex_center` is set, non-provoking vertex reads in the pixel
/// shader are lowered to center interpolation instead of requesting geometry-
/// shader vertex deltas.
#[cfg(windows)]
pub(crate) fn translate_linked(
    ms: &Shader,
    ps: Option<&Shader>,
    per_vertex_center: bool,
) -> Option<(Arc<Shader>, Option<Arc<Shader>>)> {
    static LINKED: Mutex<Option<HashMap<LinkedKey, Option<LinkedPair>>>> = Mutex::new(None);
    let key = (ms.full_hash, ps.map(|p| p.full_hash), per_vertex_center);
    if let Some(hit) = LINKED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return hit
            .as_ref()
            .map(|pair| (pair.mesh.clone(), pair.pixel.clone()));
    }
    let mesh_bytes = ms.container_bytes();
    let pixel_bytes = ps.map(Shader::container_bytes);
    let mut options = crate::shader_cache::runtime_options();
    options.per_vertex_center = per_vertex_center;
    let result = match victory_shader::emit::translate_linked(
        &mesh_bytes,
        pixel_bytes.as_deref(),
        options,
    ) {
        Ok((ms_host, ps_host)) => {
            let ms_bytes = metal_mesh_output_bytes(&ms_host.dxil);
            let ms_t = ms.translated()?;
            let mesh = ms.with_translation(Translated {
                host: ms_host,
                used: ms_t.used.clone(),
                used_written: ms_t.used_written.clone(),
                write_unknown: ms_t.write_unknown,
                threads: None,
            });
            let pixel = match (ps, ps_host) {
                (Some(p), Some(host)) => {
                    let p_t = p.translated()?;
                    Some(p.with_translation(Translated {
                        host,
                        used: p_t.used.clone(),
                        used_written: p_t.used_written.clone(),
                        write_unknown: p_t.write_unknown,
                        threads: None,
                    }))
                }
                _ => None,
            };
            tracing::info!(
                mesh = %ms.hash,
                pixel = ps.map_or("none", |p| p.hash.as_str()),
                bytes = ms_bytes,
                per_vertex_center,
                "mesh and pixel outputs linked"
            );
            Some(LinkedPair { mesh, pixel })
        }
        Err(e) => {
            tracing::warn!(
                mesh = %ms.hash,
                pixel = ps.map_or("none", |p| p.hash.as_str()),
                per_vertex_center,
                error = %e,
                "linked mesh/pixel translation failed; using standalone translation"
            );
            None
        }
    };
    let out = result
        .as_ref()
        .map(|pair| (pair.mesh.clone(), pair.pixel.clone()));
    LINKED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    out
}

/// Re-translates a pixel shader with `per_vertex_center` so a mesh pipeline
/// whose mesh stage already fits Metal's 32 KiB limit can run without
/// geometry-shader vertex deltas.
#[cfg(windows)]
pub(crate) fn translate_pixel_center(ps: &Shader) -> Option<Arc<Shader>> {
    type CenterCache = HashMap<[u8; 32], Option<Arc<Shader>>>;
    static CENTER: Mutex<Option<CenterCache>> = Mutex::new(None);
    if let Some(hit) = CENTER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.get(&ps.full_hash))
    {
        return hit.clone();
    }
    let bytes = ps.container_bytes();
    let result = if bytes.is_empty() {
        None
    } else {
        let mut options = crate::shader_cache::runtime_options();
        options.per_vertex_center = true;
        match victory_shader::emit::translate_with(&bytes, options) {
            Ok(host) => {
                let ps_t = ps.translated()?;
                let used = ps_t.used.clone();
                let (used_written, write_unknown) =
                    (ps_t.used_written.clone(), ps_t.write_unknown);
                tracing::info!(
                    pixel = %ps.hash,
                    "pixel shader re-translated with center-interpolated per-vertex fallback for mesh pipeline"
                );
                Some(ps.with_translation(Translated {
                    host,
                    used,
                    used_written,
                    write_unknown,
                    threads: ps_t.threads,
                }))
            }
            Err(e) => {
                tracing::warn!(
                    pixel = %ps.hash,
                    error = %e,
                    "pixel shader center-interpolated per-vertex fallback failed"
                );
                None
            }
        }
    };
    CENTER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(ps.full_hash, result.clone());
    result
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// Whether it is one of the title's acceleration-structure builder's
    /// ([`builder`]).
    #[cfg(windows)]
    builder: bool,
    /// What its programs are to the ambient stand-in ([`crate::ambient`]):
    /// bits, 0 for nothing (always while the stand-in is off).
    #[cfg(windows)]
    ambient: u8,
    /// Which ray pass `VICTORY_RAY_SKIP` leaves out it is
    /// ([`ray_pass_left_out`]): bits, 0 for none (always while it is off).
    #[cfg(windows)]
    ray_pass: u8,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.size, hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if it is skipped, one of its
/// shaders does not translate, the host refuses it, or it is still being made
/// (its first use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// As [`host_pipeline`], for a draw into render target and depth views of
/// the formats `bound`.
#[cfg(windows)]
pub(crate) fn host_pipeline_for<'a>(
    this: This,
    bound: &host::TargetFormats,
) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    let p = &pipeline(this).host;
    if p.formats != *bound && host::formats_differ(&p.formats, bound) {
        return make::for_formats(p, bound);
    }
    make::ready(p)
}

/// Whether a dispatch with the title pipeline is held back: it is the
/// acceleration-structure builder's, which does not run this frame
/// ([`builder::runs`]).
#[cfg(windows)]
pub(crate) fn held_back(this: This) -> bool {
    !this.is_null() && pipeline(this).builder && !builder::runs()
}

/// Whether a dispatch with the title pipeline is left out: it is one of
/// the ray passes `VICTORY_RAY_SKIP` names (`crate::ray_skip`), and the
/// acceleration-structure builder does not run this frame
/// ([`builder::runs`]), so there is nothing to trace against. Counted
/// there when it is.
#[cfg(windows)]
pub(crate) fn ray_pass_left_out(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    let kind = pipeline(this).ray_pass;
    if kind == 0 || builder::runs() {
        return false;
    }
    crate::ray_skip::left_out(kind);
    true
}

/// What the title pipeline's programs are to the ambient stand-in
/// ([`crate::ambient`]): bits, 0 for nothing.
#[cfg(windows)]
pub(crate) fn ambient_role(this: This) -> u8 {
    if this.is_null() { 0 } else { pipeline(this).ambient }
}

/// A title DLL was mapped at `base`, `size` bytes (the acceleration-
/// structure builder's pipelines are known by its image).
#[cfg(windows)]
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    builder::note_title_dll(file, base, size);
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(feature = "diagnostics", any(windows, test)))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skip_hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    })
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader<'a>(shaders: &'a [Arc<Shader>], hashes: &[String]) -> Option<&'a str> {
    if hashes.is_empty() {
        return None;
    }
    shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && pipeline(this).host.skipped_shader.is_some()
}

/// The shaders of `this`, as host pipelines are named (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shaders_of(this: This) -> String {
    if this.is_null() {
        return "no pipeline".into();
    }
    format!("{:?}", pipeline(this).shaders)
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        p.host
            .skipped_shader
            .as_deref()
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops
        .get_or_insert_with(HashMap::new)
        .entry(why)
        .or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    // SAFETY: the title passes `size` bytes of shader code.
    code_of(code).map(|bytes| shader(stage, unsafe { &*bytes }))
}

/// The bytes a stage's bytecode names, if any.
fn code_of(code: &Bytecode) -> Option<*const [u8]> {
    (!code.code.is_null() && code.size != 0)
        .then(|| std::ptr::slice_from_raw_parts(code.code, code.size))
}

/// `code`: the title's bytecode pointers (by which a pipeline of the
/// acceleration-structure builder is known, [`builder`]).
fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    code: &[*const u8],
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let builder = !code.is_empty() && builder::created_by_builder(code);
    #[cfg(not(windows))]
    let _ = code;
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(
        shaders.clone(),
        graphics,
        #[cfg(feature = "diagnostics")]
        skip_hashes(),
    ));
    // Write the pipeline down for `victory warm` (VICTORY_PIPELINE_BOOK=1).
    // Before `make::early`, so that a shader's container is still here: this
    // tree releases containers after translation.
    #[cfg(windows)]
    known::note(&make::key(&shaders, host.graphics.as_deref()), &shaders, host.graphics.as_deref());
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    if builder {
        builder::note(&host);
    }
    #[cfg(windows)]
    let ambient = crate::ambient::role(shaders.iter().map(|s| s.hash.as_str()));
    // Never a program the stand-in works after (it would not run).
    #[cfg(windows)]
    let ray_pass = if ambient == 0 {
        crate::ray_skip::kind(shaders.iter().map(|s| s.hash.as_str()))
    } else {
        0
    };
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        #[cfg(windows)]
        ambient,
        #[cfg(windows)]
        ray_pass,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        &[],
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        &[desc.stage.code],
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes a whole stream.
    let stream = unsafe { std::slice::from_raw_parts(desc.stream, desc.size) };
    #[allow(unused_variables)]
    let (shaders, graphics) = read_stream(stream, copy);
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        &[],
        iid,
        out,
    )
}

#[cfg(windows)]
type Graphics = host::GraphicsState;
#[cfg(not(windows))]
type Graphics = ();

/// A stream's shaders (each made by `shader` from its stage and bytecode)
/// and graphics state. Its pointers (shader code, input elements) must be
/// valid: the title's own, or the pre-build's.
fn read_stream(
    stream: &[u8],
    shader: impl Fn(&'static str, &Bytecode) -> Option<Arc<Shader>>,
) -> (Vec<Arc<Shader>>, Graphics) {
    let mut shaders = Vec::new();
    #[allow(unused_mut, clippy::let_unit_value)]
    let mut graphics = Graphics::default();
    let mut at = 0;
    while at + 8 <= stream.len() {
        let kind = u32::from_le_bytes(stream[at..at + 4].try_into().expect("4 bytes"));
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if at + size > stream.len() {
            break;
        }
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8, which the
            // slice does not promise to be aligned.
            let code = unsafe { stream.as_ptr().add(at + 8).cast::<Bytecode>().read_unaligned() };
            shaders.extend(shader(stage, &code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, stream.as_ptr().add(at))
        };
        at += size;
    }
    (shaders, graphics)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
mod builder;
#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
pub mod known;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;
// Also built for tests, which run where `host` cannot.
#[cfg(any(windows, test))]
mod schedule;
#[cfg(all(test, windows, feature = "diagnostics"))]
mod placed_uav_tests;
#[cfg(all(test, windows))]
mod thread_dispatch_tests;
#[cfg(all(test, windows))]
mod prewarm_cache_test;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    if std::env::var_os("VICTORY_SPILL_SHADERS").is_some_and(|v| v != "0") {
        let spill = dir.with_file_name("shader-containers");
        if crate::make_dir(&spill).is_ok() {
            tracing::info!(dir = %spill.display(), "shader containers kept on disk until translated");
            let _ = SPILL_DIR.set(spill);
        }
    }
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
    known::flush();
}

/// The pre-build: a title pipeline made ahead of a run into the pipeline
/// library (`use_pipeline_cache`), by the code a run makes it with, so the
/// run finds it there under the same name.
#[cfg(windows)]
pub struct Prebuild {
    shaders: Vec<Arc<Shader>>,
    graphics: Option<Box<host::GraphicsState>>,
}

// SAFETY: as `make::Pending`: the graphics state's input elements point
// into its own names, and it is only read.
#[cfg(windows)]
unsafe impl Send for Prebuild {}

#[cfg(windows)]
impl Prebuild {
    /// The pipeline a creation stream describes, as the title passes it to
    /// the device.
    ///
    /// # Safety
    /// The stream's pointers point at live shader code and input elements.
    pub unsafe fn graphics(stream: &[u8]) -> Prebuild {
        // Shaders of their own: a pre-build must not keep every container.
        let own = |stage, code: &Bytecode| {
            // SAFETY: the caller's pointers.
            code_of(code).map(|b| Arc::new(Shader::new(stage, unsafe { &*b })))
        };
        let (shaders, graphics) = read_stream(stream, own);
        Prebuild {
            shaders,
            graphics: Some(Box::new(graphics)),
        }
    }

    /// The compute pipeline of a container.
    pub fn compute(container: &[u8]) -> Prebuild {
        Prebuild {
            shaders: vec![Arc::new(Shader::new("compute", container))],
            graphics: None,
        }
    }

    /// What a run lists the pipeline by (its early list).
    pub fn key(&self) -> String {
        make::key(&self.shaders, self.graphics.as_deref())
    }

    /// Its shaders and state as a diagnostics build logs them.
    pub fn describe(&self) -> String {
        let state = self.graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
        format!("{:?}\t{state}", self.shaders)
    }

    /// Makes the host pipeline, or finds it in the library; why not if it
    /// gets none.
    pub fn make(&self) -> Result<(), String> {
        let key = self.key();
        library::keep_early_list();
        // Without the run's log lines per shader and pipeline: tens of
        // thousands of them would bury the pre-build's own report.
        let quiet = tracing::Dispatch::none();
        let made = tracing::dispatcher::with_default(&quiet, || {
            host::HostPipeline::new(&self.shaders, self.graphics.as_deref(), |_, _| {})
        });
        #[cfg(feature = "diagnostics")]
        make::log_built(&key);
        if made.is_some() {
            // shortcut: a pipeline the library refused to store is listed
            // too; a run then makes it at first use.
            library::note_prebuilt(key);
        }
        library::save(false);
        match made {
            Some(_) => Ok(()),
            None => Err(match self.shaders.iter().find(|s| s.translated().is_none()) {
                Some(s) => format!("{} shader does not translate", s.stage),
                None => "the host refuses it".into(),
            }),
        }
    }
}

/// Host pipelines made, and loaded from the library, since the start.
#[cfg(windows)]
pub fn pipeline_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (library::MADE.load(Relaxed), library::LOADED.load(Relaxed))
}

/// Whether a pre-build has put the title pipeline `key` in the library
/// (a second pre-build skips it without translating or loading anything).
#[cfg(windows)]
pub fn prebuilt(key: &str) -> bool {
    library::prebuilt(key)
}

/// Whether an earlier run drew or dispatched with the title pipeline `key`.
#[cfg(windows)]
pub fn wanted_early(key: &str) -> bool {
    library::wanted_early(key)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "diagnostics")]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// The slice a title's stream (or a pre-build's) is in need not be
    /// 8-aligned: its bytecode is read as it lies.
    #[test]
    fn a_stream_of_any_alignment_is_read() {
        let mut backing = [0u64; 5];
        // SAFETY: the 40 bytes of `backing`, one off its (aligned) start.
        let stream = unsafe { std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>().add(1), 32) };
        stream[..4].copy_from_slice(&1u32.to_le_bytes()); // vertex shader
        stream[8..16].copy_from_slice(&0x1000usize.to_le_bytes());
        stream[16..24].copy_from_slice(&7usize.to_le_bytes());
        assert_ne!(stream.as_ptr() as usize % 8, 0);
        let seen = std::cell::RefCell::new(Vec::new());
        read_stream(stream, |stage, code| {
            seen.borrow_mut().push((stage, code.code as usize, code.size));
            None
        });
        assert_eq!(*seen.borrow(), [("vertex", 0x1000, 7)]);
    }

    /// A spilled container leaves memory, is on disk whole, and one already
    /// there is kept rather than written again.
    #[cfg(windows)]
    #[test]
    fn spilled_containers_round_trip_through_their_file() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i * 31) as u8).collect();
        let full = *blake3::hash(&bytes).as_bytes();
        let s = Shader::new("pixel", &bytes);
        spill_to(dir.path(), &s, &full);
        assert!(s.container.lock().unwrap().is_empty(), "left memory");
        assert_eq!(s.spilled.get(), Some(&full));
        assert_eq!(crate::read_file(&spill_path(dir.path(), &full)).unwrap(), bytes);
        let again = Shader::new("pixel", &bytes);
        spill_to(dir.path(), &again, &full);
        assert!(again.container.lock().unwrap().is_empty() && again.spilled.get() == Some(&full));
    }

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.lock().unwrap().clone()))
            .collect()
    }

    /// A compute pipeline the acceleration-structure builder creates (its
    /// bytecode in the builder's image) is known as the builder's, and its
    /// dispatches are held back while its host pipeline is not made: here
    /// never, as its bytes are no shader container. Pipelines of other
    /// code are never held back.
    #[cfg(windows)]
    #[test]
    fn builder_dispatches_wait_for_every_builder_pipeline() {
        let _one = builder::test_lock();
        let code = [0xB0u8; 64];
        let create = |code: &[u8]| {
            let desc = ComputeHead {
                root: std::ptr::null(),
                stage: Bytecode {
                    code: code.as_ptr(),
                    size: code.len(),
                },
            };
            let mut p: This = std::ptr::null();
            assert_eq!(
                create_compute(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
                S_OK
            );
            p
        };
        builder::note_title_dll("xboxbvhbuilder.DLL", code.as_ptr() as usize, code.len());
        let p = create(&code);
        let other_code = [0xC0u8; 64];
        let other = create(&other_code);
        builder::note_title_dll("XboxBvhBuilder.dll", 0, 0);
        assert!(
            pipeline(p).builder,
            "its bytecode is in the builder's image"
        );
        assert!(!pipeline(other).builder);
        // Its host pipeline is started at once and refused.
        let refused = (0..500).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            make::state(&pipeline(p).host).is_some()
        });
        assert!(refused, "the builder's pipeline was started");
        assert_eq!(make::state(&pipeline(p).host), Some(false));
        assert!(
            held_back(p),
            "one builder pipeline missing: the builder waits"
        );
        assert!(!held_back(other));
        for q in [p, other] {
            slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
        }
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    #[cfg(windows)]
    #[test]
    fn translation_lets_the_container_go() {
        // Not a container: it does not translate, and goes all the same.
        let s = Shader::new("compute", &[1, 2, 3]);
        assert!(s.translated().is_none());
        assert!(s.container.lock().unwrap().is_empty());
        assert_eq!(s.size, 3);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }

    /// The pre-build reads a stream as the title's creation does: the same
    /// shaders and state, so the same key and the same host pipeline name.
    #[cfg(windows)]
    #[test]
    fn a_prebuild_is_the_pipeline_the_title_creates() {
        let (vs, ps) = ([7u8; 9], [5u8; 6]);
        let name = c"TEXCOORD";
        // One input element: name, index 2, format 16, slot 1, offset 8.
        let mut element = [0u8; 32];
        element[..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
        for (i, w) in [2u32, 16, 1, 8, 0, 0].iter().enumerate() {
            element[8 + 4 * i..12 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        let mut stream = vec![0u8; 0x2b0];
        let put = |s: &mut Vec<u8>, at: usize, v: u64| {
            s[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        let kinds = [(0x10, 1), (0x28, 2), (0x40, 3), (0x58, 4), (0x70, 5), (0x88, 8)];
        for (at, kind) in kinds.into_iter().chain([(0x210, 21), (0x250, 17)]) {
            put(&mut stream, at, kind);
        }
        put(&mut stream, 0x18, vs.as_ptr() as u64);
        put(&mut stream, 0x20, vs.len() as u64);
        put(&mut stream, 0x30, ps.as_ptr() as u64);
        put(&mut stream, 0x38, ps.len() as u64);
        put(&mut stream, 0x8c, 1 << 32); // independent blend
        put(&mut stream, 0x1d8, 9 | 0xffff_ffff << 32);
        put(&mut stream, 0x1e0, 10 | 3 << 32); // solid fill
        put(&mut stream, 0x254, 1);
        put(&mut stream, 0x260, 12);
        put(&mut stream, 0x268, element.as_ptr() as u64);
        put(&mut stream, 0x270, 1);
        put(&mut stream, 0x278, 14 | 4 << 32);
        put(&mut stream, 0x280, 15 | 87 << 32);
        put(&mut stream, 0x2a0, 1 << 32);
        put(&mut stream, 0x2a8, 16 | 40 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        assert_eq!(create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p), S_OK);
        // SAFETY: the stream points at live code and elements.
        let pre = unsafe { Prebuild::graphics(&stream) };
        let title = pipeline(p);
        assert_eq!(pre.key(), title.host.key);
        let state = title.host.graphics.as_deref().unwrap().describe();
        assert!(state.contains(r#"\"TEXCOORD\" 2 16 1 8 0 0"#), "{state}");
        assert_eq!(pre.describe(), format!("{:?}\t{state}", title.shaders));
        // Another state is another pipeline.
        put(&mut stream, 0x1e0, 10 | 2 << 32);
        // SAFETY: as above.
        assert_ne!(unsafe { Prebuild::graphics(&stream) }.key(), title.host.key);
    }

    #[cfg(windows)]
    #[test]
    fn table_uav_store_marks_only_its_own_table_entry() {
        use victory_shader::{HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};

        // Table root parameter at user_data s0 (slot 0) holding two UAVs:
        // u0 at table dword offset 0 (byte offset 0x00) and u1 at dword offset 8 (byte offset 0x20).
        let layout = vec![RootParam {
            slot: 0,
            root: RootType::Table,
            user_data: 0,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
        }];
        let used = vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 1,
                stride: 1,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ];
        let make_program = |words: &[u32]| NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: layout.clone(),
            used: used.clone(),
            io: victory_shader::native::StageIo::Cs(victory_shader::native::ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };

        // s_mov_b32 s1, 0 (high half of table address = 0, inline constant 0x80);
        // s_load_dwordx4 s[4:7], s[0:1], 0x20 (load u1 at dword offset 8 into s[4:7]);
        // buffer_store_dword v0, off, s[4:7], 0;
        // s_endpgm
        // Note: SGPR s4 does NOT match lower_bound (1), proving we trace through
        // `resource::track` rather than comparing SGPR numbers to register bounds.
        let store_u1 = make_program(&[
            0xBE81_0380, // s_mov_b32 s1, 0
            (0x3D << 26) | (2 << 18) | (4 << 6), // s_load_dwordx4 s[4:7], s[0:1], 0x20
            (125 << 25) | 0x20,
            0xE070_0000, // buffer_store_dword v0, off, s[4:7], 0
            0x8001_0000, // srsrc = 4/4 = 1, soffset = 0x80 (0)
            0xBF81_0000, // s_endpgm
        ]);
        let (written, unknown) = super::written_used(&store_u1, &store_u1.used);
        assert_eq!(written, [false, true]);
        assert!(!unknown);

        // Load-only shader touching u0 marks neither UAV as written.
        let load_only = make_program(&[
            0xBE81_0380, // s_mov_b32 s1, 0
            (0x3D << 26) | (2 << 18) | (4 << 6), // s_load_dwordx4 s[4:7], s[0:1], 0x00
            125 << 25,
            0xE030_0000, // buffer_load_dword v0, off, s[4:7], 0
            0x8001_0000,
            0xBF81_0000, // s_endpgm
        ]);
        let (written, unknown) = super::written_used(&load_only, &load_only.used);
        assert_eq!(written, [false, false]);
        assert!(!unknown);

        // Untraceable store (s[4:7] never loaded from layout) falls back conservatively.
        let untraceable = make_program(&[
            0xE070_0000,
            0x8001_0000,
            0xBF81_0000,
        ]);
        let (written, unknown) = super::written_used(&untraceable, &untraceable.used);
        assert!(unknown);
        assert_eq!(written, [true, true]);
    }

    #[test]
    fn mesh_per_vertex_center_env_toggle_defaults_on() {
        use std::ffi::OsStr;
        assert!(mesh_per_vertex_center_value(None));
        assert!(mesh_per_vertex_center_value(Some(OsStr::new("1"))));
        assert!(!mesh_per_vertex_center_value(Some(OsStr::new("0"))));
    }
}

#[cfg(all(test, windows))]
mod depth_bounds_test;
