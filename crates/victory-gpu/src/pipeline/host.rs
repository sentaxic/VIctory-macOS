//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{
    MEMORY_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE, over_limit_mesh_output_bytes,
};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR};

use super::{Shader, Translated};
use crate::backend::HostGpu;
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    pub(super) blend: D3D12_BLEND_DESC,
    pub(super) sample_mask: u32,
    pub(super) rasterizer: D3D12_RASTERIZER_DESC,
    pub(super) depth: D3D12_DEPTH_STENCIL_DESC,
    pub(super) depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    pub(super) inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    pub(super) names: Vec<CString>,
    pub(super) strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    pub(super) depth_format: DXGI_FORMAT,
    pub(super) samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

#[cfg(test)]
impl GraphicsState {
    /// A state that declares one render target, of format `format` (tests).
    pub(crate) fn declaring(format: i32) -> Self {
        let mut g = Self::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT(format);
        g
    }
}

/// The formats of what a draw draws into, as the host has them: the eight
/// render targets (0 where there is none), then depth (0 for none).
pub(crate) type TargetFormats = [u32; 9];

/// Whether a draw into views of the formats `bound` cannot go to the host
/// with a pipeline that declares `declared`: a render target slot for which
/// the pipeline declares a format has a view of another format bound to it.
pub(crate) fn formats_differ(declared: &TargetFormats, bound: &TargetFormats) -> bool {
    declared[..8]
        .iter()
        .zip(&bound[..8])
        .any(|(&d, &b)| d != 0 && b != 0 && b != d)
}

/// The formats a pipeline for a draw into `bound` declares
/// ([`formats_differ`]): the title's own, except declared render target slots
/// with a bound view take the bound view's format.
pub(crate) fn formats_for(declared: &TargetFormats, bound: &TargetFormats) -> TargetFormats {
    let mut formats = *declared;
    for i in 0..8 {
        if declared[i] != 0 && bound[i] != 0 {
            formats[i] = bound[i];
        }
    }
    formats
}

/// Integer render target formats: D3D12 cannot blend into them.
fn integer_format(f: DXGI_FORMAT) -> bool {
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// The title's blend state as the host takes it: blending off on integer
/// render targets. The console ignores blending there, and titles leave it
/// on (the world sets it on an `R8_UINT` target), but D3D12 does not allow
/// it: vkd3d-proton refuses such a pipeline with `E_INVALIDARG` ("using
/// integer format is not supported"), and its draws are lost. With one
/// description for every target, each target gets its own copy so the
/// others keep theirs.
fn host_blend(blend: &D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) -> D3D12_BLEND_DESC {
    let count = (targets.NumRenderTargets as usize).min(8);
    let per_target = blend.IndependentBlendEnable.as_bool();
    let blends = |i: usize| {
        blend.RenderTarget[if per_target { i } else { 0 }]
            .BlendEnable
            .as_bool()
    };
    if !(0..count).any(|i| integer_format(targets.RTFormats[i]) && blends(i)) {
        return *blend;
    }
    let mut host = *blend;
    if !per_target {
        host.IndependentBlendEnable = true.into();
        host.RenderTarget = [blend.RenderTarget[0]; 8];
    }
    for i in 0..count {
        if integer_format(targets.RTFormats[i]) {
            host.RenderTarget[i].BlendEnable = false.into();
        }
    }
    host
}

impl GraphicsState {
    /// The formats the state declares ([`TargetFormats`]): nothing past its
    /// count of render targets.
    pub(crate) fn formats(&self) -> TargetFormats {
        let mut formats = [0; 9];
        let n = (self.targets.NumRenderTargets as usize).min(8);
        for (to, from) in formats.iter_mut().zip(&self.targets.RTFormats[..n]) {
            *to = from.0 as u32;
        }
        formats[8] = self.depth_format.0 as u32;
        formats
    }

    /// The same state for a draw into views of the formats `bound`
    /// ([`formats_differ`]): the render target formats come from `bound`,
    /// everything else stays the title's.
    pub(crate) fn with_formats(&self, bound: &TargetFormats) -> GraphicsState {
        let names = self.names.clone();
        let inputs: Vec<D3D12_INPUT_ELEMENT_DESC> = self
            .inputs
            .iter()
            .zip(&names)
            .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                ..*e
            })
            .collect();
        let mut targets = self.targets;
        let n = (targets.NumRenderTargets as usize).min(8);
        for (to, &from) in targets.RTFormats[..n].iter_mut().zip(bound) {
            *to = DXGI_FORMAT(from as i32);
        }
        GraphicsState {
            blend: self.blend,
            sample_mask: self.sample_mask,
            rasterizer: self.rasterizer,
            depth: self.depth,
            depth_bounds: self.depth_bounds,
            inputs,
            names,
            strip_cut: self.strip_cut,
            topology: self.topology,
            targets,
            depth_format: self.depth_format,
            samples: self.samples,
        }
    }

    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    // As before, read this console subobject with the PC layout.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::target_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        let depth_bounds = if self.depth_bounds { " depth bounds" } else { "" };
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}{depth_bounds}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    fn depth_with_bounds(&self, host: &crate::host::Host) -> Option<D3D12_DEPTH_STENCIL_DESC1> {
        if !self.depth_bounds {
            return None;
        }
        if !host.depth_bounds_supported {
            crate::host::warn_depth_bounds();
            return None;
        }
        let d = self.depth;
        Some(D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: d.DepthEnable,
            DepthWriteMask: d.DepthWriteMask,
            DepthFunc: d.DepthFunc,
            StencilEnable: d.StencilEnable,
            StencilReadMask: d.StencilReadMask,
            StencilWriteMask: d.StencilWriteMask,
            FrontFace: d.FrontFace,
            BackFace: d.BackFace,
            DepthBoundsTestEnable: true.into(),
        })
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                // An appended element starts at the next multiple of
                // min(4, its size) after the previous one, where D3D12 (and
                // vkd3d) place it; the host pipeline gets the title's
                // descriptions, so the offsets read here must match.
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                        .checked_next_multiple_of(bytes.clamp(1, 4))
                        .unwrap_or(u32::MAX)
                } else {
                    e.AlignedByteOffset
                };
                next[slot] = offset.saturating_add(bytes);
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// Our root parameters of the title-memory page table and of the
    /// written-pages bitmap (`crate::memory`), if it reads or writes there.
    pub(crate) memory: Option<u32>,
    pub(crate) memory_dirty: Option<u32>,
    /// Our root parameter of the heap's view addresses (`crate::memory`).
    pub(crate) view_addresses: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// A graphics pipeline of the console's rectangle lists whose geometry
    /// shader completes each rectangle (`victory_shader::emit::rectangle_gs`):
    /// it draws them as triangles, one per rectangle.
    pub(crate) rectangles: bool,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Our root parameters of the view-mark table and the buffer views'
    /// written-pages bitmap, if a stage marks its writes through views
    /// (`victory_shader::emit`, "Write marks").
    pub(crate) marks: Option<(u32, u32)>,
}

/// The stages [`HostPipeline::new`] translates together for a mesh
/// pipeline, if any; `None` when each stage keeps its own translation.
/// Shared with [`retranslated`], so that `victory replay --retranslate`
/// reports what a run makes.
///
/// Mesh shaders whose standalone output footprint (packed or Metal-
/// aligned) exceeds Metal's 32 KiB limit are re-translated together with
/// their paired pixel shader, pruning unread outputs and repacking
/// attribute order to eliminate alignment padding holes. When the paired pixel
/// shader reads non-provoking vertex values (`VICTORY_P10`/`VICTORY_P20`),
/// mesh pipelines have no geometry stage, so lower those reads to
/// hardware center interpolation (`VICTORY_MESH_PER_VERTEX_CENTER`).
fn linked_stages(
    mesh: Option<&Arc<Shader>>,
    pixel: Option<&Arc<Shader>>,
) -> Option<(Arc<Shader>, Option<Arc<Shader>>)> {
    let ps_deltas = mesh.is_some()
        && super::mesh_per_vertex_center()
        && pixel
            .and_then(|ps| ps.translated())
            .is_some_and(|ps| victory_shader::emit::needs_vertex_deltas(&ps.host.dxil));
    mesh.and_then(|ms| {
        let over_limit = super::link_mesh_outputs()
            && over_limit_mesh_output_bytes(&ms.translated()?.host.dxil).is_some();
        if over_limit {
            super::translate_linked(ms, pixel.map(Arc::as_ref), ps_deltas)
        } else if ps_deltas {
            let ps = pixel?;
            let ps_center = super::translate_pixel_center(ps)?;
            Some((ms.clone(), Some(ps_center)))
        } else {
            None
        }
    })
}

/// For `victory replay --retranslate` (`crate::replay::play::retranslate`):
/// the DXIL each stage of a pipeline of the title's containers `stages`
/// (stage and container) gets from this build with this run's switches, as
/// [`HostPipeline::new`] chooses it ([`linked_stages`]); and whether it is
/// one made for the pipeline rather than the stage's own translation.
pub(crate) fn retranslated(stages: &[(&'static str, &[u8])]) -> Vec<(Result<Vec<u8>, String>, bool)> {
    let shaders: Vec<Arc<Shader>> = stages
        .iter()
        .map(|(stage, bytes)| super::shader(stage, bytes))
        .collect();
    let find = |stage: &str| shaders.iter().find(|s| s.stage == stage);
    let linked = linked_stages(find("mesh"), find("pixel"));
    let dxil = |s: &Arc<Shader>| {
        s.translated()
            .map(|t| t.host.dxil.clone())
            .ok_or_else(|| "it does not translate (the log says why)".to_owned())
    };
    shaders
        .iter()
        .map(|s| match (s.stage, &linked) {
            // A pixel shader lowered alone keeps the mesh shader as it is.
            ("mesh", Some((ms, _))) => (dxil(ms), !Arc::ptr_eq(ms, s)),
            ("pixel", Some((_, Some(ps)))) => (dxil(ps), true),
            _ => (dxil(s), false),
        })
        .collect()
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(
        shaders: &[Arc<Shader>],
        g: Option<&GraphicsState>,
        record_refusal: impl Fn(&'static str, Option<i32>),
    ) -> Option<HostPipeline> {
        let refuse = |reason: &'static str| {
            // Record every affected pipeline, including those after the first warning.
            record_refusal(reason, None);
            static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&reason) {
                seen.push(reason);
                tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
            }
            None
        };
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse("only vertex, mesh, pixel and compute stages are translated");
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader"),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_support().supported() {
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)");
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let linked = linked_stages(mesh, find("pixel"));
        let stages: Vec<(&Arc<Shader>, D3D12_SHADER_VISIBILITY)> = match &linked {
            Some((ms, ps)) => {
                let mut v = vec![(ms, D3D12_SHADER_VISIBILITY_MESH)];
                if let Some(ps) = ps {
                    v.push((ps, D3D12_SHADER_VISIBILITY_PIXEL));
                }
                v
            }
            None => stages,
        };
        if super::refuse_over_limit_mesh()
            && let Some((ms, _)) = stages
                .iter()
                .find(|(_, v)| *v == D3D12_SHADER_VISIBILITY_MESH)
            && let Some(t) = ms.translated()
            && let Some(bytes) = over_limit_mesh_output_bytes(&t.host.dxil)
        {
            tracing::warn!(
                ?shaders,
                bytes,
                "mesh outputs over Metal's 32 KiB after linking: pipeline refused \
                 (VICTORY_MESH_OVER_LIMIT=1 makes it)"
            );
            return refuse("the mesh shader's outputs take more than the host GPU allows");
        }
        let host = host()?;
        // `SFI0`: 64-bit atomics on typed resources and on heap resources.
        let atomics64 = stages.iter().any(|(s, _)| {
            let Some(t) = s.translated() else {
                return false;
            };
            victory_shader::Container::parse(&t.host.dxil)
                .ok()
                .and_then(|c| {
                    c.part(victory_shader::FourCC::SFI0)?
                        .data
                        .get(..4)?
                        .try_into()
                        .ok()
                })
                .is_some_and(|b: [u8; 4]| u32::from_le_bytes(b) & 0x1040_0000 != 0)
        });
        if atomics64 && !host.heap_atomics64() {
            return refuse("the host GPU has no 64-bit atomics on typed heap resources");
        }

        let aniso_point_mip_supported = crate::sampler::aniso_point_mip(&host.device);

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                memory: None,
                memory_dirty: None,
                view_addresses: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (used, kind, register, out) in [
                (t.memory, D3D12_ROOT_PARAMETER_TYPE_SRV, 0, &mut stage.memory),
                (t.memory_writes, D3D12_ROOT_PARAMETER_TYPE_UAV, 0, &mut stage.memory_dirty),
                (t.view_addresses, D3D12_ROOT_PARAMETER_TYPE_SRV, 1, &mut stage.view_addresses),
            ] {
                if used {
                    *out = Some(params.len() as u32);
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: register,
                                RegisterSpace: MEMORY_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter_for_host(aniso_point_mip_supported) as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        // One pair for all stages: they share the heap and the bitmap.
        let marks = stages
            .iter()
            .any(|(s, _)| s.translated().is_some_and(|t| t.host.view_marks))
            .then(|| {
                let first = params.len() as u32;
                for (kind, register) in [
                    (D3D12_ROOT_PARAMETER_TYPE_SRV, victory_shader::emit::VIEW_MARKS),
                    (D3D12_ROOT_PARAMETER_TYPE_UAV, victory_shader::emit::VIEW_WRITES),
                ] {
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: register,
                                RegisterSpace: MEMORY_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    });
                }
                (first, first + 1)
            });
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, compute.is_some()) {
            Ok(root) => root,
            Err(e) => {
                record_refusal("root signature refused", Some(e.code().0));
                tracing::warn!(error = %e, ?shaders, "host root signature refused");
                return None;
            }
        };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    record_refusal("compute pipeline refused", Some(e.code().0));
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
                marks,
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)");
        };
        let state = g.describe();
        let depth_with_bounds = g.depth_with_bounds(host);
        // A pixel shader reading non-provoking vertex values (#203) takes
        // them from a geometry shader made for the vertex shader.
        let ps = stages.get(1).and_then(|(s, _)| s.translated());
        let deltas = ps.is_some_and(|ps| victory_shader::emit::needs_vertex_deltas(&ps.host.dxil));
        if mesh.is_some() && deltas {
            return refuse("non-provoking vertex values in a mesh pipeline");
        }
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let ms = Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(0));
            let made = match depth_with_bounds {
                Some(depth) => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                    &name,
                ),
                None => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
                    &name,
                ),
            };
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    record_refusal("mesh pipeline refused", Some(e.code().0));
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
                marks,
            });
        }

        // Rectangle lists: a geometry shader made for the vertex shader adds
        // each rectangle's fourth corner (else rectangles are expanded on
        // the CPU from vertex inputs, see `draw`). Triangles whose pixel
        // shader reads non-provoking vertex values: one that adds them.
        let vs = stages.first().and_then(|(s, _)| s.translated());
        let gs = match (g.topology.0, vs, ps) {
            (RECTANGLES, ..) if deltas => {
                return refuse("non-provoking vertex values in a rectangle list");
            }
            (_, Some(vs), Some(ps)) if deltas => {
                if host_topology(g) != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE {
                    return refuse("non-provoking vertex values of points or lines");
                }
                match victory_shader::emit::vertex_delta_gs(&vs.host.dxil, &ps.host.dxil) {
                    Ok(gs) => gs,
                    Err(e) => {
                        record_refusal("no vertex value geometry shader", None);
                        tracing::warn!(error = %e, ?shaders, "no vertex value geometry shader");
                        return None;
                    }
                }
            }
            (RECTANGLES, Some(vs), _) => match victory_shader::emit::rectangle_gs(&vs.host.dxil) {
                Ok(gs) => gs,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "no rectangle geometry shader");
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };
        let rectangles = !gs.is_empty() && !deltas;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: if gs.is_empty() { std::ptr::null() } else { gs.as_ptr().cast() },
                BytecodeLength: gs.len(),
            },
            BlendState: host_blend(&g.blend, &g.targets),
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            PrimitiveTopologyType: host_topology(g),
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs],
            &state,
        );
        let started = std::time::Instant::now();
        let made = if let Some(depth) = depth_with_bounds {
            let vertex = VertexStream {
                vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, desc.VS),
                gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, desc.GS),
                input: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, desc.InputLayout),
                strip_cut: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
                    desc.IBStripCutValue,
                ),
            };
            stream_pipeline(
                &host.device, &root, vertex, desc.PS, g,
                Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                &name,
            )
        } else {
            super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
            )
        };
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                record_refusal("graphics pipeline refused", Some(e.code().0));
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(
            ?shaders,
            ms = started.elapsed().as_millis(),
            rectangles,
            deltas,
            "host pipeline"
        );
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            rectangles,
            slots: g.slots(),
            inputs: g.elements(),
            marks,
        })
    }
}

/// The host's topology type: the console's rectangle and quad types draw
/// as triangles on the host (see `draw`).
fn host_topology(g: &GraphicsState) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match g.topology.0 {
        RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => g.topology,
    }
}

/// A stream subobject: its type, then its description, pointer-aligned.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

/// What a vertex pipeline's stream has in place of the mesh shader (the
/// geometry shader is the rectangle lists' one, or empty).
#[repr(C)]
struct VertexStream {
    vs: Sub<D3D12_SHADER_BYTECODE>,
    gs: Sub<D3D12_SHADER_BYTECODE>,
    input: Sub<D3D12_INPUT_LAYOUT_DESC>,
    strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
}

/// The stream form used by mesh pipelines and vertex pipelines with depth bounds.
fn stream_pipeline<S, D>(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    stages: S,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    depth: Sub<D>,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    #[repr(C)]
    struct Stream<S, D> {
        root: Sub<*mut std::ffi::c_void>,
        stages: S,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        stages,
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(&g.blend, &g.targets)),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth,
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        // A vertex pipeline needs its type. A mesh pipeline's, which the
        // host derives from the mesh shader anyway, is given too: a
        // pipeline library compares descriptions with it, and loads none
        // made from a stream without it.
        topology: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, host_topology(g)),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of_val(&stream),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(formats: &[DXGI_FORMAT]) -> D3D12_RT_FORMAT_ARRAY {
        let mut t = D3D12_RT_FORMAT_ARRAY {
            NumRenderTargets: formats.len() as u32,
            ..Default::default()
        };
        t.RTFormats[..formats.len()].copy_from_slice(formats);
        t
    }

    fn blending(independent: bool) -> D3D12_BLEND_DESC {
        let mut b = GraphicsState::default().blend;
        b.IndependentBlendEnable = independent.into();
        for t in &mut b.RenderTarget {
            t.BlendEnable = true.into();
            t.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            t.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        }
        b
    }

    #[test]
    fn integer_targets_do_not_blend() {
        let t = targets(&[
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8_UINT,
        ]);
        for independent in [false, true] {
            let host = host_blend(&blending(independent), &t);
            assert!(host.IndependentBlendEnable.as_bool());
            assert!(host.RenderTarget[0].BlendEnable.as_bool());
            assert!(host.RenderTarget[1].BlendEnable.as_bool());
            assert!(!host.RenderTarget[2].BlendEnable.as_bool());
            // The rest of the target's state is the title's.
            assert_eq!(host.RenderTarget[2].SrcBlend, D3D12_BLEND_SRC_ALPHA);
            assert_eq!(host.RenderTarget[0].DestBlend, D3D12_BLEND_INV_SRC_ALPHA);
        }
    }

    #[test]
    fn blend_states_without_integer_blending_are_kept() {
        // No integer target.
        let b = blending(false);
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT]);
        assert_eq!(host_blend(&b, &t), b);
        // An integer target that does not blend.
        let mut b = blending(true);
        b.RenderTarget[1].BlendEnable = false.into();
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT]);
        assert_eq!(host_blend(&b, &t), b);
        // Integer formats past the targets in use.
        let mut t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM]);
        t.RTFormats[1] = DXGI_FORMAT_R8_UINT;
        assert_eq!(host_blend(&blending(false), &t), blending(false));
    }
    fn element(format: DXGI_FORMAT, offset: u32) -> D3D12_INPUT_ELEMENT_DESC {
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(c"TEXCOORD".as_ptr().cast()),
            SemanticIndex: 0,
            Format: format,
            InputSlot: 0,
            AlignedByteOffset: offset,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }
    }

    fn offsets(inputs: &[D3D12_INPUT_ELEMENT_DESC]) -> Vec<u32> {
        let state = GraphicsState {
            inputs: inputs.to_vec(),
            ..GraphicsState::default()
        };
        state.elements().iter().map(|e| e.offset).collect()
    }

    /// An appended element after one of 1 or 2 bytes started right after
    /// it; D3D12 aligns it to min(4, its size), so the rectangle expansion
    /// read its vertices from the wrong bytes.
    #[test]
    fn appended_elements_are_aligned() {
        const APPEND: u32 = D3D12_APPEND_ALIGNED_ELEMENT;
        // RG8 (2 bytes) then RG32 float: 4, not 2.
        let a = [
            element(DXGI_FORMAT_R8G8_UNORM, 0),
            element(DXGI_FORMAT_R32G32_FLOAT, APPEND),
        ];
        assert_eq!(offsets(&a), [0, 4]);
        // R8 then R16 float: 2, not 1; then R8 packs right after it.
        let b = [
            element(DXGI_FORMAT_R8_UNORM, 0),
            element(DXGI_FORMAT_R16_FLOAT, APPEND),
            element(DXGI_FORMAT_R8_UNORM, APPEND),
        ];
        assert_eq!(offsets(&b), [0, 2, 4]);
        // Already aligned layouts are unchanged.
        let c = [
            element(DXGI_FORMAT_R8G8B8A8_UNORM, 0),
            element(DXGI_FORMAT_R32G32_FLOAT, APPEND),
            element(DXGI_FORMAT_R32G32B32_FLOAT, APPEND),
        ];
        assert_eq!(offsets(&c), [0, 4, 12]);
    }

    /// An explicit offset near 4 GiB overflowed the next appended offset
    /// (a panic in debug builds).
    #[test]
    fn a_huge_explicit_offset_does_not_overflow() {
        let inputs = [
            element(DXGI_FORMAT_R32G32B32A32_FLOAT, u32::MAX - 8),
            element(DXGI_FORMAT_R32_FLOAT, D3D12_APPEND_ALIGNED_ELEMENT),
        ];
        let _ = offsets(&inputs);
    }

    #[test]
    fn bound_formats_are_compared_slot_by_slot() {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[0] = DXGI_FORMAT_R10G10B10A2_UNORM;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.targets.RTFormats[2] = DXGI_FORMAT_R8_UINT;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        let declared = g.formats();
        assert_eq!(declared, [24, 28, 0, 0, 0, 0, 0, 0, 40]);
        assert!(!formats_differ(&declared, &declared));
        assert!(!formats_differ(&declared, &[24, 0, 0, 0, 0, 0, 0, 0, 40]));
        assert!(!formats_differ(&declared, &[24, 28, 0, 0, 0, 0, 0, 0, 0]));
        assert!(!formats_differ(&declared, &[0; 9]));
        assert!(formats_differ(&declared, &[26, 28, 0, 0, 0, 0, 0, 0, 40]));
        assert!(!formats_differ(&declared, &[24, 28, 28, 0, 0, 0, 0, 0, 40]));
        assert!(!formats_differ(&declared, &[24, 28, 0, 0, 0, 0, 0, 0, 45]));
        assert!(!formats_differ(
            &[24, 0, 0, 0, 0, 0, 0, 0, 0],
            &[24, 0, 0, 0, 0, 0, 0, 0, 40]
        ));
        assert_eq!(
            formats_for(&declared, &[26, 28, 28, 0, 0, 0, 0, 0, 45]),
            [26, 28, 0, 0, 0, 0, 0, 0, 40]
        );
        assert_eq!(
            formats_for(&declared, &[26, 0, 0, 0, 0, 0, 0, 0, 0]),
            [26, 28, 0, 0, 0, 0, 0, 0, 40]
        );
        assert_eq!(formats_for(&declared, &declared), declared);
    }

    #[test]
    fn a_state_for_the_bound_formats() {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[0] = DXGI_FORMAT_R10G10B10A2_UNORM;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8_UINT;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        g.sample_mask = 0x55;
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.blend.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        g.names = vec![c"POSITION".to_owned()];
        g.inputs = vec![D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(g.names[0].as_ptr().cast()),
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            ..Default::default()
        }];

        let g_blend = host_blend(&g.blend, &g.targets);
        assert!(!g_blend.RenderTarget[1].BlendEnable.as_bool());

        let bound = formats_for(&g.formats(), &[26, 26, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bound, [26, 26, 0, 0, 0, 0, 0, 0, 40]);
        let v = g.with_formats(&bound);
        assert_eq!(v.formats(), bound);
        assert_eq!(v.targets.NumRenderTargets, 2);
        assert_eq!(v.depth_format, DXGI_FORMAT_D32_FLOAT);
        let v_blend = host_blend(&v.blend, &v.targets);
        assert!(v_blend.RenderTarget[0].BlendEnable.as_bool());
        assert_ne!(v.describe(), g.describe());
        assert_eq!(
            (v.sample_mask, v.rasterizer.CullMode),
            (0x55, D3D12_CULL_MODE_NONE)
        );
        assert_eq!(v.inputs.len(), 1);
        assert_eq!(v.inputs[0].SemanticName.0, v.names[0].as_ptr().cast::<u8>());
        assert_ne!(v.inputs[0].SemanticName.0, g.inputs[0].SemanticName.0);

        let bound_int = [36, 24, 0, 0, 0, 0, 0, 0, 40];
        let v_int = g.with_formats(&bound_int);
        let v_int_blend = host_blend(&v_int.blend, &v_int.targets);
        assert!(!v_int_blend.RenderTarget[0].BlendEnable.as_bool());
        assert!(v_int_blend.RenderTarget[1].BlendEnable.as_bool());
    }
}
