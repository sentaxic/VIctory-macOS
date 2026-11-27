//! Changes to a list's commands: the experiments of a replay, and lists cut
//! into pieces that are timed one by one.

use std::collections::HashSet;

use super::wire::{Barrier, Cmd, Loc, NONE};

/// Drops what runs a pipeline of `psos`; how many commands that were.
pub fn skip_pipelines(cmds: Vec<Cmd>, psos: &HashSet<u32>) -> (Vec<Cmd>, u32) {
    let mut bound = NONE;
    let mut dropped = 0;
    let out = cmds
        .into_iter()
        .filter(|c| {
            if let Cmd::Pipeline { pso } = c {
                bound = *pso;
            }
            let drop = c.runs_shaders() && psos.contains(&bound);
            dropped += u32::from(drop);
            !drop
        })
        .collect();
    (out, dropped)
}

/// Drops the copies that read or write a resource `of` names (either side
/// of a texture copy, a buffer copy or a whole one); how many.
pub fn skip_copies(cmds: Vec<Cmd>, of: impl Fn(u32) -> bool) -> (Vec<Cmd>, u32) {
    let side = |l: &Loc| match l {
        Loc::Sub { res, .. } | Loc::Placed { res, .. } => *res,
    };
    let mut dropped = 0;
    let out = cmds
        .into_iter()
        .filter(|c| {
            let drop = match c {
                Cmd::CopyTexture { dst, src, .. } => of(side(dst)) || of(side(src)),
                Cmd::CopyBuffer { dst, src, .. } | Cmd::CopyResource { dst, src } => of(*dst) || of(*src),
                _ => false,
            };
            dropped += u32::from(drop);
            !drop
        })
        .collect();
    (out, dropped)
}

/// Drops the barriers of the resources `of` names (transitions; waits for
/// unordered access, of all resources where one names none); how many.
pub fn drop_barriers(cmds: Vec<Cmd>, of: impl Fn(u32) -> bool) -> (Vec<Cmd>, u32) {
    let mut dropped = 0;
    let out = cmds
        .into_iter()
        .filter_map(|c| match c {
            Cmd::Barriers { mut all } => {
                let n = all.len();
                all.retain(|b| match b {
                    Barrier::Transition { res, .. } | Barrier::Uav { res, .. } => !of(*res),
                    Barrier::Alias { .. } => true,
                });
                dropped += (n - all.len()) as u32;
                (!all.is_empty()).then_some(Cmd::Barriers { all })
            }
            c => Some(c),
        })
        .collect();
    (out, dropped)
}

/// A buffer of the replayer's own that commands may name: the results of
/// the argument clamps, kept from a run that made them ([`keep_clamped`]).
pub const RESULTS: u32 = u32::MAX - 1;

/// A buffer of the replayer's own that [`hoist_clamps`] has clamps write
/// to and calls read from, of [`HOISTED_BYTES`].
pub const HOISTED: u32 = u32::MAX - 2;
pub const HOISTED_BYTES: u64 = 4 << 20;

/// The replayer's own pipeline, root signature and table buffer of
/// [`batch_clamps`] (`crate::indirect::make_batch`).
pub const BATCH_PSO: u32 = u32::MAX - 3;
pub const BATCH_ROOT: u32 = u32::MAX - 4;
pub const BATCH_TABLE: u32 = u32::MAX - 5;

/// The root parameters of the host layer's clamp pass (`crate::indirect`):
/// its constants (kind, stride, records, flags) and its view of where the
/// clamped arguments go.
const CONSTANTS: u32 = 0;
const DEST: u32 = 3;

/// Where the host layer's argument clamps of a frame write, from list to
/// list in the frame's order (a list of clamps alone may come before the
/// lists whose calls they serve), and how much of [`RESULTS`] is taken.
#[derive(Clone, Debug, Default)]
pub struct Clamped {
    /// By buffer and offset a clamp wrote to last: its place in
    /// [`RESULTS`] (several calls may read what one clamp wrote).
    places: std::collections::HashMap<(u32, u64), u64>,
    /// The buffers clamps wrote.
    buffers: HashSet<u32>,
    pub next: u64,
}

/// The host layer's argument clamps in a list: before a call that reads its
/// arguments from a buffer, a pass of the host layer's own copies them, cut
/// to what is safe, into an argument buffer of the list's, which the call
/// then reads. `clamps` are that pass's pipelines. Every clamp gets a place
/// of its own in [`RESULTS`]: by command, the place of each clamp's
/// dispatch and of each call that reads what a clamp wrote.
fn clamped(
    cmds: &[Cmd],
    clamps: &HashSet<u32>,
    state: &mut Clamped,
) -> std::collections::HashMap<usize, u64> {
    let mut out = std::collections::HashMap::new();
    let mut bound = NONE;
    // What the compute side has bound: where the pass writes, and how much.
    let mut dest = None;
    let mut bytes = 0;
    for (i, c) in cmds.iter().enumerate() {
        match c {
            Cmd::Pipeline { pso } => bound = *pso,
            Cmd::RootSignature {
                graphics: false, ..
            } => dest = None,
            Cmd::Constants {
                graphics: false,
                param: CONSTANTS,
                offset: 0,
                values,
            } if values.len() == 4 => bytes = u64::from(values[1]) * u64::from(values[2].max(1)),
            Cmd::RootView {
                graphics: false,
                kind: super::wire::slot::UAV,
                param: DEST,
                va: super::wire::Va::At { res, offset },
            } => dest = Some((*res, *offset)),
            Cmd::Dispatch { .. } if clamps.contains(&bound) => {
                if let Some(dest) = dest {
                    state.places.insert(dest, state.next);
                    state.buffers.insert(dest.0);
                    out.insert(i, state.next);
                    state.next += bytes.next_multiple_of(16);
                }
            }
            Cmd::Indirect { args, at, .. } => {
                if let Some(place) = state.places.get(&(*args, *at)) {
                    out.insert(i, *place);
                }
            }
            _ => {}
        }
    }
    out
}

/// The list with every clamp writing to its place in [`RESULTS`], and the
/// calls reading there: a run of it keeps what [`skip_clamped`]'s calls
/// read. (No copies of what the clamps wrote where they do: a copy with no
/// barriers of its own is not sure to run where it stands.) The frame's
/// lists in order, with one `state`.
pub fn keep_clamped(cmds: Vec<Cmd>, clamps: &HashSet<u32>, state: &mut Clamped) -> Vec<Cmd> {
    let found = clamped(&cmds, clamps, state);
    let mut out: Vec<Cmd> = Vec::with_capacity(cmds.len());
    for (i, c) in cmds.into_iter().enumerate() {
        let Some(&place) = found.get(&i) else {
            out.push(c);
            continue;
        };
        match c {
            Cmd::Indirect {
                signature,
                most,
                count,
                count_at,
                ..
            } => out.push(Cmd::Indirect {
                signature,
                most,
                args: RESULTS,
                at: place,
                count,
                count_at,
            }),
            c => {
                // The clamp's dispatch: the view it writes through, set
                // last, becomes one of its place.
                if let Some(Cmd::RootView { va, .. }) = out.iter_mut().rev().find(|c| {
                    matches!(
                        c,
                        Cmd::RootView {
                            graphics: false,
                            param: DEST,
                            ..
                        }
                    )
                }) {
                    *va = super::wire::Va::At {
                        res: RESULTS,
                        offset: place,
                    };
                }
                out.push(c);
            }
        }
    }
    out
}

/// What the host layer would record had it hoisted a list's clamps (as it
/// does for runs of indirect draws, `crate::indirect::bound_before`): every
/// clamp that the host layer recorded in place, just before the indirect
/// draw it serves, with its three barriers, moves to the start of the list,
/// where it writes to a place of its own in [`HOISTED`] (from `next` on, as
/// far as [`HOISTED_BYTES`]), and its draw reads there. Only clamps with
/// nothing before them in the list that may write (a dispatch, copy, clear,
/// resolve or immediate write of the title's, or an indirect dispatch) move,
/// and only those of draws (`dispatches`: the command signatures that
/// dispatch): what the clamp reads is then as it was where the clamp stood.
/// The list and how many clamps moved.
pub fn hoist_clamps(
    cmds: Vec<Cmd>,
    clamps: &HashSet<u32>,
    roots: &HashSet<u32>,
    dispatches: &HashSet<u32>,
    next: &mut u64,
) -> (Vec<Cmd>, u32) {
    // The clamps that move: the range of commands that bind and run each
    // (from its root signature to its dispatch), the buffer and offset it
    // writes, its place, and the call it serves.
    struct Moved {
        from: usize,
        to: usize,
        dest: (u32, u64),
        place: u64,
        call: usize,
    }
    let mut moved: Vec<Moved> = Vec::new();
    let mut bound = NONE;
    let mut wrote = false;
    let mut start = None;
    let (mut dest, mut bytes) = (None, 0u64);
    for (i, c) in cmds.iter().enumerate() {
        match c {
            Cmd::Pipeline { pso } => bound = *pso,
            Cmd::RootSignature {
                graphics: false,
                root,
            } => {
                start = roots.contains(root).then_some(i);
                dest = None;
            }
            Cmd::Constants {
                graphics: false,
                param: CONSTANTS,
                offset: 0,
                values,
            } if values.len() == 4 => bytes = u64::from(values[1]) * u64::from(values[2].max(1)),
            Cmd::RootView {
                graphics: false,
                kind: super::wire::slot::UAV,
                param: DEST,
                va: super::wire::Va::At { res, offset },
            } => dest = Some((*res, *offset)),
            Cmd::Dispatch { .. } if clamps.contains(&bound) => {
                let (Some(from), Some(d)) = (start, dest) else {
                    wrote = true;
                    continue;
                };
                // In place: the call it serves comes before anything else
                // that works, in this list.
                let call = cmds[i + 1..]
                    .iter()
                    .position(|c| c.works())
                    .map(|k| i + 1 + k)
                    .filter(|&k| matches!(&cmds[k], Cmd::Indirect { args, at, signature, .. } if (*args, *at) == d && !dispatches.contains(signature)));
                let room = bytes.next_multiple_of(16);
                match call {
                    Some(call) if !wrote && *next + room <= HOISTED_BYTES => {
                        moved.push(Moved {
                            from,
                            to: i,
                            dest: d,
                            place: *next,
                            call,
                        });
                        *next += room;
                    }
                    _ => wrote = true,
                }
            }
            Cmd::Indirect { signature, .. } if dispatches.contains(signature) => wrote = true,
            Cmd::Dispatch { .. }
            | Cmd::CopyBuffer { .. }
            | Cmd::CopyTexture { .. }
            | Cmd::CopyResource { .. }
            | Cmd::Resolve { .. }
            | Cmd::ResolveRegion { .. }
            | Cmd::ResolveQuery { .. }
            | Cmd::ClearDepth { .. }
            | Cmd::ClearTarget { .. }
            | Cmd::Discard { .. }
            | Cmd::WriteImmediate { .. } => wrote = true,
            _ => {}
        }
    }
    if moved.is_empty() {
        return (cmds, 0);
    }
    let n = moved.len() as u32;
    let transition = |before: i32, after: i32| Barrier::Transition {
        flags: 0,
        res: HOISTED,
        sub: u32::MAX,
        before,
        after,
    };
    let wait = Barrier::Uav { flags: 0, res: NONE };
    // The clamps first, one after another, as the host layer's hoisted
    // ones run; then the buffer becomes one of arguments.
    let mut out: Vec<Cmd> = vec![Cmd::Barriers {
        all: vec![transition(0, 8)],
    }];
    for m in &moved {
        out.push(Cmd::Barriers { all: vec![wait.clone()] });
        for c in &cmds[m.from..=m.to] {
            out.push(match c {
                Cmd::RootView {
                    graphics: false,
                    kind,
                    param: DEST,
                    ..
                } => Cmd::RootView {
                    graphics: false,
                    kind: *kind,
                    param: DEST,
                    va: super::wire::Va::At {
                        res: HOISTED,
                        offset: m.place,
                    },
                },
                c => c.clone(),
            });
        }
    }
    out.push(Cmd::Barriers {
        all: vec![wait, transition(8, 0x200)],
    });
    // The rest without them: their commands, the barriers of the buffers
    // they wrote, and their calls reading their places.
    let dests: HashSet<u32> = moved.iter().map(|m| m.dest.0).collect();
    let mut skip = vec![false; cmds.len()];
    for m in &moved {
        skip[m.from..=m.to].iter_mut().for_each(|s| *s = true);
    }
    let calls: std::collections::HashMap<usize, u64> = moved.iter().map(|m| (m.call, m.place)).collect();
    // Whether a barrier command stands next to a moved clamp or call.
    let near: HashSet<usize> = moved
        .iter()
        .flat_map(|m| [m.from.wrapping_sub(1), m.to + 1, m.call + 1])
        .collect();
    for (i, c) in cmds.into_iter().enumerate() {
        if skip[i] {
            continue;
        }
        match c {
            Cmd::Indirect {
                signature,
                most,
                count,
                count_at,
                ..
            } if calls.contains_key(&i) => out.push(Cmd::Indirect {
                signature,
                most,
                args: HOISTED,
                at: calls[&i],
                count,
                count_at,
            }),
            Cmd::Barriers { mut all } if near.contains(&i) => {
                all.retain(|b| match b {
                    Barrier::Transition { res, .. } => !dests.contains(res),
                    Barrier::Uav { res, .. } => *res != NONE,
                    Barrier::Alias { .. } => true,
                });
                if !all.is_empty() {
                    out.push(Cmd::Barriers { all });
                }
            }
            c => out.push(c),
        }
    }
    out.push(Cmd::Barriers {
        all: vec![transition(0x200, 0)],
    });
    (out, n)
}

/// One of the host layer's hoisted clamps: what it binds and runs.
struct Hoisted {
    kind: u32,
    stride: u32,
    records: u32,
    flags: u32,
    args: (u32, u64),
    count: (u32, u64),
    dest: (u32, u64),
    budget: u32,
}

/// The list with each run of the host layer's hoisted clamps (a clamp list
/// before the list that draws: each clamp an unordered-access barrier, its
/// binding and its dispatch, one after another) replaced by batches
/// (`crate::indirect`'s batch program): one dispatch that walks the clamps
/// of the run in order, writing the same results and charging the same
/// budget. A batch ends where the run does, where its clamps' destination
/// or budget changes, or where they would read more than
/// `crate::batch::BATCH_SOURCES` buffers. Its table goes to `table`
/// ([`BATCH_TABLE`] at word offsets). Barriers between the clamps that only
/// make their destinations writable go before the batch. The list, how
/// many batches and how many clamps went into them.
pub fn batch_clamps(
    cmds: Vec<Cmd>,
    clamps: &HashSet<u32>,
    roots: &HashSet<u32>,
    table: &mut Vec<u32>,
) -> (Vec<Cmd>, u32, u32) {
    use super::wire::{Va, slot};
    use crate::batch::{BATCH_SOURCES, Entry};
    let wait = |c: &Cmd| matches!(c, Cmd::Barriers { all } if all.iter().all(|b| matches!(b, Barrier::Uav { res: NONE, .. })));
    let writable = |c: &Cmd| {
        matches!(c, Cmd::Barriers { all } if all.iter().all(|b| matches!(b, Barrier::Uav { res: NONE, .. } | Barrier::Transition { before: 0, after: 8, .. })))
    };
    // A clamp at `i` (its root signature): what it is, and the command
    // after its dispatch.
    let block = |i: usize| -> Option<(Hoisted, usize)> {
        let Cmd::RootSignature { graphics: false, root } = cmds.get(i)? else {
            return None;
        };
        if !roots.contains(root) {
            return None;
        }
        let mut h = Hoisted {
            kind: 0,
            stride: 0,
            records: 0,
            flags: 0,
            args: (NONE, 0),
            count: (NONE, 0),
            dest: (NONE, 0),
            budget: NONE,
        };
        let mut piped = false;
        for (k, c) in cmds.iter().enumerate().skip(i + 1) {
            match c {
                Cmd::Pipeline { pso } => piped = clamps.contains(pso),
                Cmd::Constants { graphics: false, param: CONSTANTS, offset: 0, values } if values.len() == 4 => {
                    (h.kind, h.stride, h.records, h.flags) = (values[0], values[1], values[2], values[3]);
                }
                Cmd::RootView { graphics: false, param, va: Va::At { res, offset }, .. } => match param {
                    1 => h.args = (*res, *offset),
                    2 | 5 => h.budget = *res,
                    3 => h.dest = (*res, *offset),
                    4 => h.count = (*res, *offset),
                    _ => return None,
                },
                Cmd::Dispatch { .. } if piped && h.args.0 != NONE && h.dest.0 != NONE && h.budget != NONE => {
                    return Some((h, k + 1));
                }
                _ => return None,
            }
        }
        None
    };
    let mut out = Vec::with_capacity(cmds.len());
    let (mut batches, mut batched) = (0, 0);
    let mut i = 0;
    while i < cmds.len() {
        // A run starts with a wait for earlier writes and a clamp.
        let mut run: Vec<Hoisted> = Vec::new();
        let mut before: Vec<Cmd> = Vec::new();
        let mut at = i;
        loop {
            let mut k = at;
            let mut skipped = Vec::new();
            while k < cmds.len() && writable(&cmds[k]) && !(wait(&cmds[k]) && block(k + 1).is_some()) {
                skipped.push(cmds[k].clone());
                k += 1;
            }
            if k < cmds.len() && wait(&cmds[k])
                && let Some((h, next)) = block(k + 1)
            {
                before.extend(skipped);
                run.push(h);
                at = next;
                continue;
            }
            break;
        }
        let fits = |v: u64| u32::try_from(v).is_ok();
        if run.len() < 2 || !run.iter().all(|h| fits(h.args.1) && fits(h.count.1) && fits(h.dest.1)) {
            out.push(cmds[i].clone());
            i += 1;
            continue;
        }
        out.extend(before);
        // In batches of one destination, budget and at most so many
        // sources.
        let mut rest = &run[..];
        while !rest.is_empty() {
            let mut sources: Vec<u32> = Vec::new();
            let mut n = 0;
            for h in rest {
                let mut wanted = sources.clone();
                for res in [h.args.0, if h.flags & 1 != 0 { h.count.0 } else { h.args.0 }] {
                    if !wanted.contains(&res) {
                        wanted.push(res);
                    }
                }
                if wanted.len() > BATCH_SOURCES || h.dest.0 != rest[0].dest.0 || h.budget != rest[0].budget {
                    break;
                }
                sources = wanted;
                n += 1;
            }
            // (A clamp reads two sources at the most: every batch takes one.)
            let (now, after) = rest.split_at(n);
            let slot = |res: u32| sources.iter().position(|s| *s == res).unwrap_or(0) as u32;
            let first = table.len() as u64 * 4;
            for h in now {
                let counted = h.flags & 1 != 0;
                let e = Entry {
                    kind: h.kind,
                    stride: h.stride,
                    records: h.records,
                    flags: h.flags,
                    from: (slot(h.args.0), h.args.1 as u32),
                    count: if counted { (slot(h.count.0), h.count.1 as u32) } else { (slot(h.args.0), 0) },
                    to: h.dest.1 as u32,
                };
                table.extend(e.words());
            }
            out.push(Cmd::Barriers { all: vec![Barrier::Uav { flags: 0, res: NONE }] });
            out.push(Cmd::RootSignature { graphics: false, root: BATCH_ROOT });
            out.push(Cmd::Pipeline { pso: BATCH_PSO });
            out.push(Cmd::Constants { graphics: false, param: 0, offset: 0, values: vec![now.len() as u32] });
            for k in 0..BATCH_SOURCES {
                let res = sources.get(k).or(sources.first()).copied().unwrap_or(now[0].args.0);
                out.push(Cmd::RootView { graphics: false, kind: slot::SRV, param: 1 + k as u32, va: Va::At { res, offset: 0 } });
            }
            out.push(Cmd::RootView { graphics: false, kind: slot::SRV, param: 5, va: Va::At { res: BATCH_TABLE, offset: first } });
            out.push(Cmd::RootView { graphics: false, kind: slot::UAV, param: 6, va: Va::At { res: now[0].dest.0, offset: 0 } });
            out.push(Cmd::RootView { graphics: false, kind: slot::UAV, param: 7, va: Va::At { res: now[0].budget, offset: 0 } });
            out.push(Cmd::RootView { graphics: false, kind: slot::UAV, param: 8, va: Va::At { res: now[0].budget, offset: 0 } });
            out.push(Cmd::Dispatch { groups: [1, 1, 1] });
            batches += 1;
            batched += now.len() as u32;
            rest = after;
        }
        i = at;
    }
    (out, batches, batched)
}

/// The list without its argument clamps: the clamp passes (with what binds
/// them, and the barriers of the buffers they write) are left out, and the
/// calls they served read the arguments kept in [`RESULTS`]. How many
/// passes and how many barriers were left out. The frame's lists in order,
/// with one `state`.
pub fn skip_clamped(
    cmds: Vec<Cmd>,
    clamps: &HashSet<u32>,
    roots: &HashSet<u32>,
    state: &mut Clamped,
) -> (Vec<Cmd>, u32, u32) {
    let found = clamped(&cmds, clamps, state);
    let buffers = state.buffers.clone();
    let (mut passes, mut barriers) = (0, 0);
    let mut out: Vec<Cmd> = Vec::with_capacity(cmds.len());
    // Where the binding of a clamp pass began, if one is being bound.
    let mut binding: Option<usize> = None;
    let mut bound = NONE;
    for (i, c) in cmds.into_iter().enumerate() {
        match &c {
            Cmd::RootSignature {
                graphics: false,
                root,
            } => binding = roots.contains(root).then_some(out.len()),
            Cmd::Pipeline { pso } => bound = *pso,
            Cmd::Constants { .. } | Cmd::RootView { .. } | Cmd::Table { .. } => {}
            Cmd::Dispatch { .. } if clamps.contains(&bound) => {
                // The pass and what bound it.
                if let Some(from) = binding.take() {
                    out.truncate(from);
                }
                passes += 1;
                continue;
            }
            Cmd::Barriers { all } => {
                // The barriers of a clamp's buffer, and the wait for all
                // unordered access that came with one.
                if all.iter().any(
                    |b| matches!(b, Barrier::Transition { res, .. } if buffers.contains(res)),
                ) {
                    let left: Vec<Barrier> = all
                        .iter()
                        .filter(|b| match b {
                            Barrier::Transition { res, .. } => !buffers.contains(res),
                            Barrier::Uav { res, .. } => *res != NONE,
                            Barrier::Alias { .. } => true,
                        })
                        .cloned()
                        .collect();
                    barriers += (all.len() - left.len()) as u32;
                    if !left.is_empty() {
                        out.push(Cmd::Barriers { all: left });
                    }
                    continue;
                }
            }
            _ => binding = None,
        }
        if let Some(&from) = found.get(&i)
            && let Cmd::Indirect {
                signature,
                most,
                count,
                count_at,
                ..
            } = &c
        {
            out.push(Cmd::Indirect {
                signature: *signature,
                most: *most,
                args: RESULTS,
                at: from,
                count: *count,
                count_at: *count_at,
            });
            continue;
        }
        out.push(c);
    }
    (out, passes, barriers)
}

/// The list with every scissor rectangle one texel: draws cover next to
/// nothing, and what is left of their time is not the pixels'.
pub fn one_texel(cmds: Vec<Cmd>) -> Vec<Cmd> {
    cmds.into_iter()
        .map(|c| match c {
            Cmd::Scissors { all } => Cmd::Scissors {
                all: all.iter().map(|r| [r[0], r[1], r[0] + 1, r[1] + 1]).collect(),
            },
            c => c,
        })
        .collect()
}

/// The list with every dispatch of one thread group.
pub fn one_group(cmds: Vec<Cmd>) -> Vec<Cmd> {
    cmds.into_iter()
        .map(|c| match c {
            Cmd::Dispatch { .. } => Cmd::Dispatch { groups: [1, 1, 1] },
            c => c,
        })
        .collect()
}

/// The list without the commands that work: what it binds and its barriers.
pub fn binds_only(cmds: Vec<Cmd>) -> Vec<Cmd> {
    cmds.into_iter().filter(|c| !c.works()).collect()
}

/// What a command that works is counted as.
pub fn kind(c: &Cmd, dispatches: &HashSet<u32>, mesh: &HashSet<u32>, bound: u32) -> &'static str {
    match c {
        Cmd::Barriers { .. } => "barriers",
        Cmd::Dispatch { .. } => "dispatches",
        Cmd::Indirect { signature, .. } if dispatches.contains(signature) => "indirect dispatches",
        Cmd::Indirect { .. } if mesh.contains(&bound) => "indirect mesh draws",
        Cmd::Indirect { .. } => "indirect draws",
        Cmd::DispatchMesh { .. } => "mesh draws",
        Cmd::Draw { .. } | Cmd::DrawIndexed { .. } => "draws",
        Cmd::ClearDepth { .. } | Cmd::ClearTarget { .. } | Cmd::Discard { .. } => "clears",
        Cmd::Resolve { .. } | Cmd::ResolveRegion { .. } => "resolves",
        Cmd::CopyBuffer { .. } => "copies of buffers",
        Cmd::CopyTexture {
            dst: Loc::Placed { .. },
            ..
        } => "copies of textures to buffers",
        Cmd::CopyTexture {
            src: Loc::Placed { .. },
            ..
        } => "copies of buffers to textures",
        Cmd::CopyTexture { .. } => "copies of textures",
        Cmd::CopyResource { .. } => "copies of whole resources",
        _ => "other",
    }
}

/// What a list has bound: the commands that bind it again, in an order that
/// holds (heaps, then root signatures before their arguments).
#[derive(Clone, Debug, Default)]
pub struct Bound {
    heaps: Option<Cmd>,
    rest: Vec<Cmd>,
}

impl Bound {
    /// Takes note of a command; false for one that binds nothing (it works,
    /// or changes resources' states).
    pub fn note(&mut self, c: &Cmd) -> bool {
        // What a command stands in the place of: an earlier one of its key.
        let same = |a: &Cmd, b: &Cmd| match (a, b) {
            (Cmd::Pipeline { .. }, Cmd::Pipeline { .. })
            | (Cmd::Topology { .. }, Cmd::Topology { .. })
            | (Cmd::Viewports { .. }, Cmd::Viewports { .. })
            | (Cmd::Scissors { .. }, Cmd::Scissors { .. })
            | (Cmd::BlendFactor { .. }, Cmd::BlendFactor { .. })
            | (Cmd::StencilRef { .. }, Cmd::StencilRef { .. })
            | (Cmd::DepthBounds { .. }, Cmd::DepthBounds { .. })
            | (Cmd::IndexBuffer { .. }, Cmd::IndexBuffer { .. })
            | (Cmd::Targets { .. }, Cmd::Targets { .. }) => true,
            (
                Cmd::RootSignature { graphics: a, .. },
                Cmd::RootSignature { graphics: b, .. },
            ) => a == b,
            (
                Cmd::Table {
                    graphics: a,
                    param: p,
                    ..
                },
                Cmd::Table {
                    graphics: b,
                    param: q,
                    ..
                },
            )
            | (
                Cmd::RootView {
                    graphics: a,
                    param: p,
                    ..
                },
                Cmd::RootView {
                    graphics: b,
                    param: q,
                    ..
                },
            ) => (a, p) == (b, q),
            (
                Cmd::Constants {
                    graphics: a,
                    param: p,
                    offset: o,
                    values: v,
                },
                Cmd::Constants {
                    graphics: b,
                    param: q,
                    offset: n,
                    values: w,
                },
            ) => (a, p, o, v.len()) == (b, q, n, w.len()),
            (
                Cmd::VertexBuffers {
                    start: a, count: n, ..
                },
                Cmd::VertexBuffers {
                    start: b, count: m, ..
                },
            ) => (a, n) == (b, m),
            _ => false,
        };
        match c {
            Cmd::Heaps { .. } => self.heaps = Some(c.clone()),
            Cmd::RootSignature { graphics, .. } => {
                // A new root signature has no arguments yet.
                self.rest.retain(|old| {
                    !matches!(
                        old,
                        Cmd::RootSignature { graphics: g, .. }
                        | Cmd::Table { graphics: g, .. }
                        | Cmd::Constants { graphics: g, .. }
                        | Cmd::RootView { graphics: g, .. } if g == graphics
                    )
                });
                self.rest.push(c.clone());
            }
            Cmd::Pipeline { .. }
            | Cmd::Topology { .. }
            | Cmd::Viewports { .. }
            | Cmd::Scissors { .. }
            | Cmd::BlendFactor { .. }
            | Cmd::StencilRef { .. }
            | Cmd::DepthBounds { .. }
            | Cmd::IndexBuffer { .. }
            | Cmd::Targets { .. }
            | Cmd::Table { .. }
            | Cmd::RootView { .. }
            | Cmd::Constants { .. }
            | Cmd::VertexBuffers { .. } => {
                self.rest.retain(|old| !same(old, c));
                self.rest.push(c.clone());
            }
            _ => return false,
        }
        true
    }

    /// The commands that bind again what is bound.
    pub fn again(&self) -> impl Iterator<Item = &Cmd> {
        self.heaps.iter().chain(&self.rest)
    }
}

/// A piece of a list: commands that can run as a list of their own.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub cmds: Vec<Cmd>,
    /// What its work is counted as, and the pipeline it runs ([`NONE`] for
    /// work that runs none).
    pub kind: &'static str,
    pub pso: u32,
    /// Its commands that work.
    pub works: u32,
}

/// Cuts a list into pieces: each holds the commands that work of one kind
/// with one pipeline bound, one after the other, with what is bound at its
/// start bound again in front and the barriers and bindings between them
/// in their places. Run in order, the pieces do what the list does.
pub fn pieces(cmds: &[Cmd], dispatches: &HashSet<u32>, mesh: &HashSet<u32>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut bound = Bound::default();
    let mut pso = NONE;
    // What came since the last command that works, and what was bound
    // before it.
    let mut since: Vec<Cmd> = Vec::new();
    let mut before = bound.clone();
    for c in cmds {
        if let Cmd::Pipeline { pso: p } = c {
            pso = *p;
        }
        let works = c.works();
        if !works {
            bound.note(c);
            since.push(c.clone());
            continue;
        }
        let runs = if c.runs_shaders() { pso } else { NONE };
        let k = kind(c, dispatches, mesh, pso);
        match out.last_mut() {
            Some(last) if last.kind == k && last.pso == runs => {
                last.cmds.append(&mut since);
                last.cmds.push(c.clone());
                last.works += 1;
            }
            _ => {
                let mut all: Vec<Cmd> = before.again().cloned().collect();
                all.append(&mut since);
                all.push(c.clone());
                out.push(Piece {
                    cmds: all,
                    kind: k,
                    pso: runs,
                    works: 1,
                });
            }
        }
        before = bound.clone();
    }
    // What follows the last work (barriers, mostly) still has to run.
    if !since.is_empty() {
        match out.last_mut() {
            Some(last) => last.cmds.append(&mut since),
            None => out.push(Piece {
                cmds: since,
                kind: "other",
                pso: NONE,
                works: 0,
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::wire::{Va, slot};

    fn pipeline(pso: u32) -> Cmd {
        Cmd::Pipeline { pso }
    }

    fn draw(n: u32) -> Cmd {
        Cmd::Draw {
            vertices: n,
            instances: 1,
            first: 0,
            first_instance: 0,
        }
    }

    fn barrier(res: u32) -> Cmd {
        Cmd::Barriers {
            all: vec![Barrier::Uav { flags: 0, res }],
        }
    }

    fn root(graphics: bool, root: u32) -> Cmd {
        Cmd::RootSignature { graphics, root }
    }

    fn table(graphics: bool, param: u32, slot: u32) -> Cmd {
        Cmd::Table {
            graphics,
            param,
            heap: 1,
            slot,
        }
    }

    #[test]
    fn pipelines_are_skipped() {
        let copy = Cmd::CopyResource { dst: 1, src: 2 };
        let cmds = vec![
            pipeline(1),
            draw(3),
            pipeline(2),
            draw(4),
            copy.clone(),
            Cmd::Dispatch { groups: [1, 1, 1] },
            pipeline(1),
            draw(5),
        ];
        let (left, dropped) = skip_pipelines(cmds, &HashSet::from([2]));
        assert_eq!(dropped, 2);
        assert_eq!(
            left,
            [pipeline(1), draw(3), pipeline(2), copy, pipeline(1), draw(5)]
        );
    }

    #[test]
    fn copies_are_skipped() {
        let placed = |res| Loc::Placed {
            res,
            offset: 0,
            format: 28,
            size: [4, 4, 1],
            pitch: 256,
        };
        let sub = |res| Loc::Sub { res, sub: 0 };
        let texture = |dst, src| Cmd::CopyTexture {
            dst,
            at: [0; 3],
            src,
            part: None,
        };
        let cmds = vec![
            texture(placed(7), sub(1)),
            texture(sub(2), placed(7)),
            texture(sub(2), placed(8)),
            Cmd::CopyBuffer {
                dst: 9,
                dst_at: 0,
                src: 7,
                src_at: 0,
                len: 4,
            },
            Cmd::CopyResource { dst: 2, src: 1 },
            draw(3),
        ];
        let (left, dropped) = skip_copies(cmds.clone(), |id| id == 7);
        assert_eq!(dropped, 3);
        assert_eq!(left, [cmds[2].clone(), cmds[4].clone(), draw(3)]);
    }

    #[test]
    fn barriers_are_dropped() {
        let transition = |res| Barrier::Transition {
            flags: 0,
            res,
            sub: 0,
            before: 0,
            after: 8,
        };
        let cmds = vec![
            Cmd::Barriers {
                all: vec![transition(1), transition(2), Barrier::Uav { flags: 0, res: 1 }],
            },
            draw(3),
            barrier(1),
            Cmd::Barriers {
                all: vec![Barrier::Alias {
                    flags: 0,
                    before: 1,
                    after: 2,
                }],
            },
        ];
        let (left, dropped) = drop_barriers(cmds.clone(), |id| id == 1);
        assert_eq!(dropped, 3);
        assert_eq!(
            left,
            [
                Cmd::Barriers {
                    all: vec![transition(2)]
                },
                draw(3),
                cmds[3].clone()
            ]
        );
        assert_eq!(drop_barriers(cmds, |_| true).1, 4);
    }

    #[test]
    fn clamps_are_kept_and_skipped() {
        let (clamp, clamp_root) = (9, 8);
        let view = |param, res, offset| Cmd::RootView {
            graphics: false,
            kind: slot::UAV,
            param,
            va: Va::At { res, offset },
        };
        // Two records of 12 bytes.
        let constants = Cmd::Constants {
            graphics: false,
            param: 0,
            offset: 0,
            values: vec![2, 12, 2, 0],
        };
        let transition = |res, before, after| Cmd::Barriers {
            all: vec![
                Barrier::Uav { flags: 0, res: NONE },
                Barrier::Transition {
                    flags: 0,
                    res,
                    sub: u32::MAX,
                    before,
                    after,
                },
            ],
        };
        let call = |args, at| Cmd::Indirect {
            signature: 5,
            most: 2,
            args,
            at,
            count: NONE,
            count_at: 0,
        };
        let dispatch = Cmd::Dispatch { groups: [1, 1, 1] };
        let cmds = vec![
            // The title's own pass, and a call it feeds unclamped.
            root(false, 1),
            pipeline(1),
            view(3, 40, 0),
            dispatch.clone(),
            call(40, 0),
            // A clamped call.
            transition(50, 0, 8),
            root(false, clamp_root),
            pipeline(clamp),
            constants.clone(),
            view(2, 70, 0),
            view(3, 50, 16),
            dispatch.clone(),
            transition(50, 8, 0x200),
            root(false, 1),
            pipeline(1),
            call(50, 16),
            transition(50, 0x200, 0),
            // Another call that reads what that clamp wrote.
            call(50, 16),
        ];
        let clamps = HashSet::from([clamp]);
        let mut state = Clamped {
            next: 96,
            ..Default::default()
        };
        // Kept: the clamp writes to its place, and its calls read there.
        let kept = keep_clamped(cmds.clone(), &clamps, &mut state);
        assert_eq!(state.next, 128);
        let mut expected = cmds.clone();
        expected[10] = view(3, RESULTS, 96);
        expected[15] = call(RESULTS, 96);
        expected[17] = call(RESULTS, 96);
        assert_eq!(kept, expected);
        let mut state = Clamped {
            next: 96,
            ..Default::default()
        };
        let roots = HashSet::from([clamp_root]);
        let (left, passes, barriers) = skip_clamped(cmds.clone(), &clamps, &roots, &mut state);
        assert_eq!((passes, barriers, state.next), (1, 6, 128));
        assert_eq!(
            left,
            [
                cmds[0].clone(),
                cmds[1].clone(),
                cmds[2].clone(),
                dispatch.clone(),
                call(40, 0),
                root(false, 1),
                pipeline(1),
                call(RESULTS, 96),
                call(RESULTS, 96),
            ]
        );
        // A list of clamps alone, and the calls they serve in the next one;
        // a place written twice is read as it was written last.
        let ahead: Vec<Cmd> = [0, 24, 0]
            .into_iter()
            .flat_map(|at| {
                [
                    root(false, clamp_root),
                    pipeline(clamp),
                    constants.clone(),
                    view(3, 60, at),
                    dispatch.clone(),
                ]
            })
            .collect();
        let calls = vec![call(60, 24), call(60, 0), call(60, 48)];
        let mut state = Clamped::default();
        let first = keep_clamped(ahead.clone(), &clamps, &mut state);
        assert_eq!(first[3], view(3, RESULTS, 0));
        assert_eq!(first[8], view(3, RESULTS, 32));
        assert_eq!(first[13], view(3, RESULTS, 64));
        let second = keep_clamped(calls.clone(), &clamps, &mut state);
        assert_eq!(second, [call(RESULTS, 32), call(RESULTS, 64), call(60, 48)]);
        let mut state = Clamped::default();
        let (first, passes, _) = skip_clamped(ahead, &clamps, &roots, &mut state);
        assert_eq!((first.len(), passes), (0, 3));
        let (second, _, _) = skip_clamped(calls, &clamps, &roots, &mut state);
        assert_eq!(second, [call(RESULTS, 32), call(RESULTS, 64), call(60, 48)]);
    }

    #[test]
    fn clamps_in_place_are_hoisted() {
        let (clamp, clamp_root) = (9, 8);
        let view = |param, res, offset| Cmd::RootView {
            graphics: false,
            kind: slot::UAV,
            param,
            va: Va::At { res, offset },
        };
        let constants = Cmd::Constants {
            graphics: false,
            param: 0,
            offset: 0,
            values: vec![0, 16, 1, 2],
        };
        let barrier = |res, before, after, wait: bool| Cmd::Barriers {
            all: wait
                .then_some(Barrier::Uav { flags: 0, res: NONE })
                .into_iter()
                .chain([Barrier::Transition {
                    flags: 0,
                    res,
                    sub: u32::MAX,
                    before,
                    after,
                }])
                .collect(),
        };
        let call = |signature, args, at| Cmd::Indirect {
            signature,
            most: 1,
            args,
            at,
            count: NONE,
            count_at: 0,
        };
        // A clamp in place before the call of signature `signature`, and the
        // draw pipeline bound again: as `crate::indirect::bound` records.
        let in_place = |signature, src| {
            vec![
                barrier(50, 0, 8, true),
                root(false, clamp_root),
                pipeline(clamp),
                constants.clone(),
                view(1, src, 0),
                view(3, 50, 0),
                Cmd::Dispatch { groups: [1, 1, 1] },
                barrier(50, 8, 0x200, true),
                root(true, 2),
                pipeline(3),
                call(signature, 50, 0),
                barrier(50, 0x200, 0, false),
            ]
        };
        let draws = 5;
        let dispatches = HashSet::from([6]);
        let mut cmds = vec![root(true, 2), pipeline(3), draw(3)];
        cmds.extend(in_place(draws, 60));
        cmds.extend(in_place(draws, 61));
        // An indirect dispatch's clamp stays; so does every clamp after it.
        cmds.extend(in_place(dispatches.iter().copied().next().unwrap(), 62));
        cmds.extend(in_place(draws, 63));
        let mut next = 32;
        let (out, n) = hoist_clamps(cmds.clone(), &HashSet::from([clamp]), &HashSet::from([clamp_root]), &dispatches, &mut next);
        assert_eq!((n, next), (2, 64));
        // The two clamps first, writing to their places.
        let hoisted = |at| Va::At { res: HOISTED, offset: at };
        let views: Vec<&Cmd> = out.iter().filter(|c| matches!(c, Cmd::RootView { param: 3, .. })).collect();
        assert_eq!(views[0], &Cmd::RootView { graphics: false, kind: slot::UAV, param: 3, va: hoisted(32) });
        assert_eq!(views[1], &Cmd::RootView { graphics: false, kind: slot::UAV, param: 3, va: hoisted(48) });
        assert_eq!(views[2..], [&view(3, 50, 0), &view(3, 50, 0)]);
        let first_draw = out.iter().position(|c| c == &draw(3)).unwrap();
        let dispatched = out.iter().filter(|c| matches!(c, Cmd::Dispatch { .. })).count();
        assert_eq!(dispatched, 4);
        assert!(out[..first_draw].iter().filter(|c| matches!(c, Cmd::Dispatch { .. })).count() == 2);
        // Their calls read there, with no barrier of theirs between the draws.
        let calls: Vec<&Cmd> = out.iter().filter(|c| matches!(c, Cmd::Indirect { .. })).collect();
        assert_eq!(calls, [&call(draws, HOISTED, 32), &call(draws, HOISTED, 48), &call(6, 50, 0), &call(draws, 50, 0)]);
        let between = &out[first_draw..out.iter().position(|c| c == calls[1]).unwrap()];
        assert!(!between.iter().any(|c| matches!(c, Cmd::Barriers { .. } | Cmd::Dispatch { .. })), "{between:?}");
        // The buffer is common at the list's end.
        assert_eq!(out.last(), Some(&Cmd::Barriers { all: vec![Barrier::Transition { flags: 0, res: HOISTED, sub: u32::MAX, before: 0x200, after: 0 }] }));
        // Nothing moves where the list wrote first.
        let mut wrote = vec![Cmd::Dispatch { groups: [1, 1, 1] }];
        wrote.extend(in_place(draws, 60));
        let mut next = 0;
        let (same, n) = hoist_clamps(wrote.clone(), &HashSet::from([clamp]), &HashSet::from([clamp_root]), &dispatches, &mut next);
        assert_eq!((n, next), (0, 0));
        assert_eq!(same, wrote);
    }

    #[test]
    fn hoisted_clamps_are_batched() {
        let (clamp, clamp_root) = (9, 8);
        let view = |kind, param, res, offset| Cmd::RootView {
            graphics: false,
            kind,
            param,
            va: Va::At { res, offset },
        };
        let wait = Cmd::Barriers {
            all: vec![Barrier::Uav { flags: 0, res: NONE }],
        };
        // A hoisted clamp of records at `src`, into the run's buffer 70 at
        // `to`, counted (by the word at 61:8) or not.
        let hoisted = |src: (u32, u64), to, counted: bool| {
            vec![
                wait.clone(),
                root(false, clamp_root),
                pipeline(clamp),
                Cmd::Constants {
                    graphics: false,
                    param: 0,
                    offset: 0,
                    values: vec![10, 12, 3, 2 | u32::from(counted)],
                },
                view(slot::SRV, 1, src.0, src.1),
                view(slot::UAV, 2, 50, 0),
                view(slot::UAV, 3, 70, to),
                view(slot::SRV, 4, if counted { 61 } else { src.0 }, if counted { 8 } else { src.1 }),
                view(slot::UAV, 5, 50, 0),
                Cmd::Dispatch { groups: [1, 1, 1] },
            ]
        };
        let mut cmds = vec![draw(3)];
        cmds.extend(hoisted((60, 0), 0, false));
        // The destination made writable between two clamps.
        cmds.push(Cmd::Barriers {
            all: vec![Barrier::Transition { flags: 0, res: 70, sub: u32::MAX, before: 0, after: 8 }],
        });
        cmds.extend(hoisted((60, 32), 48, true));
        // Four sources more: the fifth starts a second batch.
        for (i, src) in [62, 63, 64].into_iter().enumerate() {
            cmds.extend(hoisted((src, 0), 96 + 48 * i as u64, false));
        }
        cmds.push(draw(4));
        // A clamp alone is left as it is.
        let single = hoisted((60, 64), 400, false);
        cmds.extend(single.clone());
        let mut table = vec![7; 3];
        let (out, batches, batched) = batch_clamps(cmds.clone(), &HashSet::from([clamp]), &HashSet::from([clamp_root]), &mut table);
        assert_eq!((batches, batched), (2, 5));
        // Sources 60, 61 (the count), 62, 63 in the first; 64 in the second.
        assert_eq!(table.len(), 3 + 5 * crate::batch::ENTRY);
        let entry = |i: usize| &table[3 + i * crate::batch::ENTRY..3 + (i + 1) * crate::batch::ENTRY];
        assert_eq!(entry(0), [10, 12, 3, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(entry(1), [10, 12, 3, 3, 0, 32, 1, 8, 48, 0]);
        assert_eq!(entry(2), [10, 12, 3, 2, 2, 0, 2, 0, 96, 0]);
        assert_eq!(entry(3), [10, 12, 3, 2, 3, 0, 3, 0, 144, 0]);
        assert_eq!(entry(4), [10, 12, 3, 2, 0, 0, 0, 0, 192, 0]);
        let dispatches: Vec<usize> = out.iter().enumerate().filter(|c| matches!(c.1, Cmd::Dispatch { .. })).map(|c| c.0).collect();
        assert_eq!(dispatches.len(), 3);
        assert_eq!(out[dispatches[0] - 4], view(slot::SRV, 5, BATCH_TABLE, 12));
        assert_eq!(out[dispatches[1] - 4], view(slot::SRV, 5, BATCH_TABLE, 12 + 4 * 4 * crate::batch::ENTRY as u64));
        assert_eq!(out[dispatches[0] - 3], view(slot::UAV, 6, 70, 0));
        assert_eq!(out[dispatches[0] - 2], view(slot::UAV, 7, 50, 0));
        assert_eq!(out[dispatches[0] - 1], view(slot::UAV, 8, 50, 0));
        // The barrier between the clamps goes before the first batch.
        assert_eq!(out[1], cmds[cmds.iter().position(|c| matches!(c, Cmd::Barriers { all } if all.len() == 1 && matches!(all[0], Barrier::Transition { .. }))).unwrap()]);
        assert!(out.ends_with(&single));
        assert_eq!(out[0], draw(3));
    }

    #[test]
    fn draws_without_pixels_and_lists_without_work() {
        let cmds = vec![
            Cmd::Scissors {
                all: vec![[10, 20, 300, 400]],
            },
            pipeline(1),
            draw(3),
            barrier(1),
        ];
        assert_eq!(
            one_texel(cmds.clone())[0],
            Cmd::Scissors {
                all: vec![[10, 20, 11, 21]]
            }
        );
        assert_eq!(
            binds_only(cmds.clone()),
            [cmds[0].clone(), pipeline(1), barrier(1)]
        );
    }

    #[test]
    fn what_is_bound_is_bound_again_in_order() {
        let mut b = Bound::default();
        let heaps = Cmd::Heaps { heaps: vec![1, 2] };
        let constants = |offset, values: Vec<u32>| Cmd::Constants {
            graphics: true,
            param: 2,
            offset,
            values,
        };
        let view = |param, offset| Cmd::RootView {
            graphics: false,
            kind: slot::CBV,
            param,
            va: Va::At { res: 3, offset },
        };
        for c in [
            root(true, 1),
            table(true, 0, 10),
            pipeline(7),
            heaps.clone(),
            table(true, 0, 11),
            table(true, 1, 12),
            constants(0, vec![1, 2]),
            constants(2, vec![3]),
            constants(0, vec![4, 5]),
            root(false, 2),
            view(0, 64),
            view(0, 128),
            pipeline(8),
        ] {
            assert!(b.note(&c));
        }
        assert!(!b.note(&draw(3)) && !b.note(&barrier(1)));
        let again: Vec<Cmd> = b.again().cloned().collect();
        assert_eq!(
            again,
            [
                heaps,
                root(true, 1),
                table(true, 0, 11),
                table(true, 1, 12),
                constants(2, vec![3]),
                constants(0, vec![4, 5]),
                root(false, 2),
                view(0, 128),
                pipeline(8),
            ]
        );
        // A new root signature drops its side's arguments, and only those.
        b.note(&root(true, 3));
        let again: Vec<Cmd> = b.again().cloned().collect();
        assert_eq!(
            again[1..],
            [root(false, 2), view(0, 128), pipeline(8), root(true, 3)]
        );
    }

    #[test]
    fn a_list_in_pieces_does_what_the_list_does() {
        let none = HashSet::new();
        let copy = Cmd::CopyResource { dst: 1, src: 2 };
        let cmds = vec![
            root(true, 1),
            pipeline(1),
            table(true, 0, 5),
            draw(1),
            table(true, 0, 6),
            draw(2),
            barrier(9),
            pipeline(2),
            draw(3),
            copy.clone(),
            draw(4),
            barrier(8),
        ];
        let all = pieces(&cmds, &none, &none);
        let shape: Vec<(&str, u32, u32)> = all.iter().map(|p| (p.kind, p.pso, p.works)).collect();
        assert_eq!(
            shape,
            [
                ("draws", 1, 2),
                ("draws", 2, 1),
                ("copies of whole resources", NONE, 1),
                ("draws", 2, 1)
            ]
        );
        assert_eq!(all[0].cmds, cmds[..6]);
        // The second starts with what the first had bound at its end; the
        // barrier and the new pipeline stand where they stood.
        assert_eq!(
            all[1].cmds,
            [
                root(true, 1),
                pipeline(1),
                table(true, 0, 6),
                barrier(9),
                pipeline(2),
                draw(3)
            ]
        );
        assert_eq!(
            all[2].cmds,
            [root(true, 1), table(true, 0, 6), pipeline(2), copy]
        );
        // What follows the last work stays with it.
        assert_eq!(all[3].cmds.last(), Some(&barrier(8)));
        // Every command that works or changes states is there once, in order.
        let once = |all: &[Cmd]| -> Vec<Cmd> {
            all.iter()
                .filter(|c| c.works() || matches!(c, Cmd::Barriers { .. }))
                .cloned()
                .collect()
        };
        let joined: Vec<Cmd> = all.iter().flat_map(|p| p.cmds.clone()).collect();
        assert_eq!(once(&joined), once(&cmds));
        // A list of barriers alone is one piece.
        let only = pieces(&[barrier(1)], &none, &none);
        assert_eq!((only.len(), only[0].works), (1, 0));
        assert!(pieces(&[], &none, &none).is_empty());
    }
}
