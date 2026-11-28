//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{
    BufferBinding, HeapIndexSource, HostShader, ScalarSource, StaticSampler, UserData,
};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder
/// of this translator, `None` if it cannot be opened.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    let dir = dir.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(&dir) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "shader cache");
            let _ = CACHE.set(cache);
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            None
        }
    }
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What tells one set of translations from another ([`build`]). The warming
/// of host pipelines keeps it beside them, so that a change of the
/// translator makes the warmed ones stale.
pub(crate) fn translation() -> [u8; 32] {
    *build()
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings (the loop limit, which a diagnostics build can
/// change at run time, and whether the host's waves are 32 lanes at most,
/// [`host_wave32`]).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(
            victory_shader::SOURCE_HASH,
            ENCODING,
            limit,
            host_wave32(),
            victory_shader::emit::thread_ticket(),
        )
    })
}

/// Whether translations are for a host of 32-lane waves: the host GPU's
/// (made here if it is not yet: the cache belongs to the GPU it was filled
/// on, as the pipeline library does); none without one.
fn host_wave32() -> bool {
    crate::host::host().is_some_and(crate::backend::HostGpu::wave32)
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(
    translator: &str,
    encoding: &[u8],
    limit: u32,
    wave32: bool,
    thread_ticket: bool,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    h.update(&[u8::from(wave32), u8::from(thread_ticket)]);
    *h.finalize().as_bytes()
}

/// Whether translations put 64-bit image atomics under locks: under Wine,
/// whose D3DMetal claims the atomics and leaves the texels untouched.
/// `VICTORY_ATOMIC_LOCKS=0` or `1` overrides.
fn atomic_locks() -> bool {
    static LOCKS: OnceLock<bool> = OnceLock::new();
    *LOCKS.get_or_init(|| match std::env::var("VICTORY_ATOMIC_LOCKS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether the lanes of a wave first try their 64-bit atomics' locks in
/// rounds before the lanes still not done take turns (`emit`, "Atomic
/// locks"; #240): wherever [`atomic_locks`] are on. In the game the rounds
/// took a world frame from 7.9 to 2.5 s on an M2 Max, from 5.29 to 0.71 s
/// on an M4 Pro and from 2.9 to 0.88 s on an M4, and left no GPU resets.
/// `VICTORY_ATOMIC_LOCK_ROUNDS=0` goes back to the turns alone.
fn atomic_lock_rounds() -> bool {
    static ROUNDS: OnceLock<bool> = OnceLock::new();
    *ROUNDS.get_or_init(|| {
        let var = std::env::var("VICTORY_ATOMIC_LOCK_ROUNDS").ok();
        rounds_on(atomic_locks(), var.as_deref())
    })
}

/// Whether a glc-clear 64-bit `umax` first reduces each texel's lanes of
/// a wave to their maximum key, one representative locking (`emit`,
/// "Atomic locks"; #240): opt-in while it is measured,
/// `VICTORY_ATOMIC_UMAX=1`. Only with [`atomic_lock_rounds`].
fn atomic_umax() -> bool {
    static UMAX: OnceLock<bool> = OnceLock::new();
    *UMAX.get_or_init(|| {
        atomic_lock_rounds() && std::env::var("VICTORY_ATOMIC_UMAX").as_deref() == Ok("1")
    })
}

/// The rounds, given whether the locks are on and
/// `VICTORY_ATOMIC_LOCK_ROUNDS`: on with the locks unless it is `0`.
fn rounds_on(locks: bool, var: Option<&str>) -> bool {
    locks && var != Some("0")
}

/// Whether compute shaders' biased samples run as scaled gradient samples:
/// under Wine, whose D3DMetal ignores the bias there (#30).
/// `VICTORY_BIAS_AS_GRADIENTS=0` or `1` overrides.
fn bias_as_gradients() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_BIAS_AS_GRADIENTS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether translations take wave64 programs to run in 32-lane host waves:
/// under Wine, whose D3DMetal reports 32 lanes and ignores `WaveSize(64)`.
/// `VICTORY_NARROW_WAVES=0` or `1` overrides. Also what the offline
/// `translate --wave32` follows, so its results match `run`'s.
pub fn narrow_waves() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_NARROW_WAVES").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => under_wine(),
    })
}

/// Whether the two 32-lane halves of a console wave64 pass each other
/// values in divergent code by waiting for each other on group-shared
/// memory (`victory_shader::emit::Options::halves_rendezvous`): on, wherever
/// programs run in halves. `VICTORY_HALVES_RENDEZVOUS=0` goes back to
/// refusing such programs.
fn halves_rendezvous() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_HALVES_RENDEZVOUS").as_deref() != Ok("0"))
}

/// Whether 64-thread wave64 shaders keep only the per-lane translation.
fn literal_halves_disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("VICTORY_LITERAL_HALVES").as_deref() == Ok("0"))
}

/// Whether a size query on a writable image asks for level 0 in every
/// program (`Options::uav_size_level0`): only with
/// `VICTORY_UAV_SIZE_LEVEL0=1`. D3DMetal answers the undefined level DXC
/// writes with a size of 0, so a program that computes with that size
/// goes wrong: the title's shadow min/max program (`07f451a14b13fecb`)
/// writes one flat value and sun shadows get 16x16-tile edges. Level 0
/// in every program also changed the TAA disocclusion mask (speckles on
/// skin), so a list of shader hashes instead of `1` turns it on for those
/// programs only ([`uav_size_level0_for`]).
fn uav_size_level0() -> bool {
    uav_size_rule().0
}

/// Whether the program in `container` gets the level-0 size query: with
/// `VICTORY_UAV_SIZE_LEVEL0=1`, or when the variable lists its hash (16
/// digits, as the log prints shader hashes; several with commas).
fn uav_size_level0_for(container: &[u8]) -> bool {
    let (all, only) = uav_size_rule();
    *all || (!only.is_empty() && only.contains(&blake3::hash(container).to_hex()[..16].to_owned()))
}

fn uav_size_rule() -> &'static (bool, Vec<String>) {
    static RULE: OnceLock<(bool, Vec<String>)> = OnceLock::new();
    RULE.get_or_init(|| {
        let rule = uav_size(std::env::var("VICTORY_UAV_SIZE_LEVEL0").ok().as_deref());
        if rule.0 {
            tracing::info!("VICTORY_UAV_SIZE_LEVEL0=1: size queries on writable images ask for level 0");
        } else if !rule.1.is_empty() {
            tracing::info!(
                programs = ?rule.1,
                "VICTORY_UAV_SIZE_LEVEL0: size queries on writable images ask for level 0 in these programs only"
            );
        }
        rule
    })
}

/// `VICTORY_UAV_SIZE_LEVEL0`'s rule: `1` is every program; a list of
/// 16-digit shader hashes is those programs; anything else is off.
fn uav_size(value: Option<&str>) -> (bool, Vec<String>) {
    let value = value.unwrap_or_default().trim();
    if value == "1" {
        return (true, Vec::new());
    }
    let hashes: Vec<String> = value
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .collect();
    if hashes
        .iter()
        .all(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        (false, hashes)
    } else {
        (false, Vec::new())
    }
}

/// Whether this runs under Wine (on a Mac: D3DMetal's D3D12).
#[cfg(windows)]
pub(crate) fn under_wine() -> bool {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: ntdll is always loaded; both names end with a NUL.
    unsafe {
        GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_version".as_ptr().cast(),
        )
    }
    .is_some()
}

#[cfg(not(windows))]
pub(crate) fn under_wine() -> bool {
    false
}

/// [`build`] for translations with host workarounds (atomic locks, their
/// rounds and umax reduction, bias as gradients, narrow waves, level-0
/// size queries on writable images, and literal halves): their own cache
/// entries, one per combination.
pub(crate) fn translation_build(options: victory_shader::emit::Options) -> &'static [u8; 32] {
    let locks = options.atomic_locks;
    let bias = options.bias_as_gradients;
    let narrow = options.narrow_waves;
    let rounds = locks && options.atomic_lock_rounds;
    let umax = rounds && options.atomic_umax;
    let uav_level0 = options.uav_size_level0;
    let no_literal = options.host_wave32 && options.disable_literal_halves;
    // Only where programs run in halves does the rendezvous change them.
    let apart = options.host_wave32 && !options.halves_rendezvous;
    let cap_barriers = victory_shader::emit::cap_barriers_refused();
    let limits = options.thread_limits;
    if !locks
        && !bias
        && !narrow
        && !no_literal
        && !uav_level0
        && !apart
        && !cap_barriers
        && !limits
    {
        return build();
    }
    static BUILDS: [OnceLock<[u8; 32]>; 1024] = [const { OnceLock::new() }; 1024];
    let n = usize::from(locks)
        | usize::from(bias) << 1
        | usize::from(narrow) << 2
        | usize::from(rounds) << 3
        | usize::from(umax) << 4
        | usize::from(no_literal) << 5
        | usize::from(uav_level0) << 6
        | usize::from(apart) << 7
        | usize::from(cap_barriers) << 8
        | usize::from(limits) << 9;
    BUILDS[n].get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(build());
        if locks {
            h.update(b"atomic locks");
        }
        if rounds {
            h.update(b"atomic lock rounds");
        }
        if umax {
            h.update(b"atomic umax");
        }
        if bias {
            h.update(b"bias as gradients");
        }
        if narrow {
            h.update(b"narrow waves");
        }
        if uav_level0 {
            h.update(b"uav size level 0");
        }
        if no_literal {
            h.update(b"literal halves disabled");
        }
        if apart {
            h.update(b"no halves rendezvous");
        }
        if cap_barriers {
            h.update(b"cap barriers refused");
        }
        if limits {
            h.update(b"thread limits");
        }
        *h.finalize().as_bytes()
    })
}

pub(crate) fn runtime_options() -> victory_shader::emit::Options {
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    // Under Wine (D3DMetal), 64-bit image atomics run under locks: its own
    // leave the texels untouched (`emit`, "Atomic locks"). On a host of
    // 32-lane waves, wave64 programs run in halves (`build` keys on it).
    victory_shader::emit::Options {
        view_marks: true,
        atomic_locks: atomic_locks(),
        atomic_lock_rounds: atomic_lock_rounds(),
        atomic_umax: atomic_umax(),
        bias_as_gradients: bias_as_gradients(),
        host_wave32: host_wave32(),
        narrow_waves: narrow_waves(),
        disable_literal_halves: literal_halves_disabled(),
        per_vertex_center: false,
        uav_size_level0: uav_size_level0(),
        halves_rendezvous: halves_rendezvous(),
        // Only for the pipelines a dispatch counted in threads needs it
        // for (`crate::pipeline::thread_limited`).
        thread_limits: false,
    }
}

/// The translation of a console shader container, from the cache when
/// there is one, and whether it came from there.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    host_shader_with(container, runtime_options())
}

/// [`host_shader`] made as `options` say (cached under their own key).
pub(crate) fn host_shader_with(
    container: &[u8],
    options: victory_shader::emit::Options,
) -> (Result<HostShader, String>, bool) {
    let mut options = options;
    // One program at a time, if the switch lists hashes.
    options.uav_size_level0 = uav_size_level0_for(container);
    let translate =
        |c: &[u8]| victory_shader::emit::translate_with(c, options).map_err(|e| e.to_string());
    let build = translation_build(options);
    match CACHE.get() {
        Some(cache) => cached(cache, build, container, translate),
        None => (translate(container), false),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit;
/// `made` keeps a miss's translation, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms here).
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let made = (self.translate)(&input[32..]);
        let bytes = encode(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return (shader, false);
        }
        if let Some(shader) = crate::read_file(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    (
        made().unwrap_or_else(|| (converter.translate)(container)),
        false,
    )
}

const MAGIC: &[u8; 4] = b"VSH3";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        heap_index_sources,
        view_marks,
        half_views,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, root_srv_bytes.len() as u32);
    for &(slot, end) in root_srv_bytes {
        word(&mut out, u32::from(slot));
        word(&mut out, end);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    let encode_binding = |out: &mut Vec<u8>, b: BufferBinding| match b {
        BufferBinding::Table {
            slot,
            table_offset_dwords,
        } => {
            word(out, 0);
            word(out, u32::from(slot));
            word(out, table_offset_dwords);
        }
        BufferBinding::RootDescriptor { slot } => {
            word(out, 1);
            word(out, u32::from(slot));
        }
    };
    let encode_scalar = |out: &mut Vec<u8>, sc: ScalarSource| match sc {
        ScalarSource::Const(v) => {
            word(out, 0);
            word(out, v);
        }
        ScalarSource::BufferWord {
            binding,
            byte_offset,
        } => {
            word(out, 1);
            encode_binding(out, binding);
            word(out, byte_offset);
        }
        ScalarSource::RootConstant { slot, dword } => {
            word(out, 2);
            word(out, u32::from(slot));
            word(out, dword);
        }
    };
    word(&mut out, heap_index_sources.len() as u32);
    for src in heap_index_sources {
        match *src {
            HeapIndexSource::CbvWord {
                binding,
                byte_offset,
                index_offset,
            } => {
                word(&mut out, 0);
                encode_binding(&mut out, binding);
                word(&mut out, byte_offset);
                word(&mut out, index_offset);
            }
            HeapIndexSource::StructuredLoopField {
                buffer,
                stride,
                field_byte_offset,
                start,
                end,
                end_is_count,
                inclusive_extra,
                index_offset,
            } => {
                word(&mut out, 1);
                encode_binding(&mut out, buffer);
                word(&mut out, stride.unwrap_or(0));
                word(&mut out, field_byte_offset);
                encode_scalar(&mut out, start);
                encode_scalar(&mut out, end);
                word(&mut out, u32::from(end_is_count));
                word(&mut out, inclusive_extra);
                word(&mut out, index_offset);
            }
        }
    }
    word(&mut out, half_views.len() as u32);
    for &(slot, dword) in half_views {
        word(&mut out, u32::from(slot));
        word(&mut out, dword);
    }
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_srv_bytes = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_srv_bytes.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    let memory = r.word()?;
    let decode_binding = |r: &mut Reader<'_>| -> Option<BufferBinding> {
        match r.word()? {
            0 => Some(BufferBinding::Table {
                slot: u8::try_from(r.word()?).ok()?,
                table_offset_dwords: r.word()?,
            }),
            1 => Some(BufferBinding::RootDescriptor {
                slot: u8::try_from(r.word()?).ok()?,
            }),
            _ => None,
        }
    };
    let decode_scalar = |r: &mut Reader<'_>| -> Option<ScalarSource> {
        match r.word()? {
            0 => Some(ScalarSource::Const(r.word()?)),
            1 => Some(ScalarSource::BufferWord {
                binding: decode_binding(r)?,
                byte_offset: r.word()?,
            }),
            2 => Some(ScalarSource::RootConstant {
                slot: u8::try_from(r.word()?).ok()?,
                dword: r.word()?,
            }),
            _ => None,
        }
    };
    let n = r.word()? as usize;
    let mut heap_index_sources = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let src = match r.word()? {
            0 => HeapIndexSource::CbvWord {
                binding: decode_binding(&mut r)?,
                byte_offset: r.word()?,
                index_offset: r.word()?,
            },
            1 => {
                let buffer = decode_binding(&mut r)?;
                let stride = match r.word()? {
                    0 => None,
                    s => Some(s),
                };
                let field_byte_offset = r.word()?;
                let start = decode_scalar(&mut r)?;
                let end = decode_scalar(&mut r)?;
                let end_is_count = match r.word()? {
                    0 => false,
                    1 => true,
                    _ => return None,
                };
                let inclusive_extra = r.word()?;
                let index_offset = r.word()?;
                HeapIndexSource::StructuredLoopField {
                    buffer,
                    stride,
                    field_byte_offset,
                    start,
                    end,
                    end_is_count,
                    inclusive_extra,
                    index_offset,
                }
            }
            _ => return None,
        };
        heap_index_sources.push(src);
    }
    let n = r.word()? as usize;
    let mut half_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        half_views.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        heap_index_sources,
        view_marks: memory & 32 != 0,
        half_views,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounds are on wherever the locks are, `VICTORY_ATOMIC_LOCK_ROUNDS=0`
    /// turns them off, and without the locks there are none (#240).
    #[test]
    fn lock_rounds_are_on_with_the_locks_unless_turned_off() {
        assert!(rounds_on(true, None), "on by default with the locks");
        assert!(rounds_on(true, Some("1")), "the old opt-in still works");
        assert!(!rounds_on(true, Some("0")), "0 goes back to the turns");
        assert!(!rounds_on(false, None), "no rounds without the locks");
        assert!(!rounds_on(false, Some("1")), "nor when asked for");
    }

    #[test]
    fn the_size_query_switch_is_all_programs_or_a_list_of_hashes() {
        assert_eq!(uav_size(None), (false, vec![]));
        assert_eq!(uav_size(Some("")), (false, vec![]));
        assert_eq!(uav_size(Some("0")), (false, vec![]));
        assert_eq!(uav_size(Some("1")), (true, vec![]));
        assert_eq!(uav_size(Some("yes")), (false, vec![]));
        let two = vec!["07f451a14b13fecb".to_owned(), "11eeb27c2d15980d".to_owned()];
        assert_eq!(
            uav_size(Some("07F451A14B13FECB, 11eeb27c2d15980d")),
            (false, two)
        );
        // A list with a wrong entry is no list: nothing is turned on.
        assert_eq!(uav_size(Some("07f451a14b13fecb,1")), (false, vec![]));
    }

    /// Translations with the level-0 size query have cache entries of
    /// their own.
    #[test]
    fn the_size_query_switch_keys_the_cache() {
        let plain = victory_shader::emit::Options::default();
        let level0 = victory_shader::emit::Options {
            uav_size_level0: true,
            ..plain
        };
        assert_eq!(translation_build(plain), build());
        assert_ne!(translation_build(level0), build());
    }

    #[test]
    fn disabling_literal_halves_changes_the_translation_key() {
        let options = victory_shader::emit::Options {
            host_wave32: true,
            ..runtime_options()
        };
        let on = translation_build(victory_shader::emit::Options {
            disable_literal_halves: false,
            ..options
        });
        let off = translation_build(victory_shader::emit::Options {
            disable_literal_halves: true,
            ..options
        });
        assert_ne!(on, off);
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            root_srv_bytes: vec![(1, 48), (5, 16384)],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            heap_index_sources: vec![
                HeapIndexSource::CbvWord {
                    binding: BufferBinding::Table {
                        slot: 1,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 16,
                    index_offset: 0,
                },
                HeapIndexSource::StructuredLoopField {
                    buffer: BufferBinding::Table {
                        slot: 1,
                        table_offset_dwords: 8,
                    },
                    stride: Some(64),
                    field_byte_offset: 12,
                    start: ScalarSource::BufferWord {
                        binding: BufferBinding::Table {
                            slot: 1,
                            table_offset_dwords: 0,
                        },
                        byte_offset: 4,
                    },
                    end: ScalarSource::BufferWord {
                        binding: BufferBinding::Table {
                            slot: 1,
                            table_offset_dwords: 0,
                        },
                        byte_offset: 8,
                    },
                    end_is_count: false,
                    inclusive_extra: 1,
                    index_offset: 0,
                },
            ],
            view_marks: true,
            half_views: vec![(0, 8), (4, 0)],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &[7; 32], b"a container", translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the host's wave width) makes
    /// another key, so no entry written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let k = key("0123456789abcdef", b"encoding", 1 << 20, false, true);
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20, false, true));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20, false, true));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20, false, true));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19, false, true));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 20, true, true));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 20, false, false));
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                host_wave32(),
                victory_shader::emit::thread_ticket(),
            )
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
