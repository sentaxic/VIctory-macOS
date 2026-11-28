//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
#[cfg(windows)]
mod prebuild;
#[cfg(windows)]
mod replay;
mod run;
mod shader;
#[cfg(windows)]
mod translate;
#[cfg(windows)]
mod warm;

use std::num::NonZeroUsize;
#[cfg(any(windows, test))]
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout. Shader packs
    /// among the paths are listed last, their stages named from the
    /// containers found.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found (and per pack pipeline).
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
        /// Instead list the effects of shader databases: technique, pass,
        /// shaders (name, container hash, engine hash) and pass state.
        #[arg(long)]
        effects: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader. With
        /// `--cache`: then make every pipeline of the title's shader pack
        /// into the host pipeline library `run` reads (on this GPU and
        /// driver; a second time only what is not there yet).
        #[arg(long)]
        pipelines: bool,
        /// With `--cache`, pre-build only pack pipelines named in this file
        /// (one full key per line, as in early-pipelines.txt).
        #[arg(long, requires = "cache", conflicts_with = "pipelines")]
        pipeline_selection: Option<PathBuf>,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Translate for a host whose waves have at most 32 lanes (NVIDIA,
        /// Intel), as `run` does there: wave64 compute programs in halves.
        /// Under Wine (D3DMetal) also as `run` does there: a group that is
        /// one wave64 with LDS gets barriers between its halves
        /// (`VICTORY_NARROW_WAVES=0` or `1` overrides). With `--pipelines`,
        /// set by the GPU that makes them.
        #[arg(long, conflicts_with = "cache")]
        wave32: bool,
        /// Positive worker thread count (default: every logical core).
        #[arg(long)]
        jobs: Option<NonZeroUsize>,
        /// Only translate into the shader cache `run` reads (the warm-up):
        /// every container found, once, nothing else checked or listed
        /// (with `--pipelines`, then the shader pack's pipelines).
        #[arg(long, conflicts_with_all = ["validate", "list"])]
        cache: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Run a captured frame's GPU work again without the title, with what
    /// it costs and experiments on it (docs/replay.md).
    #[cfg(windows)]
    Replay(Box<replay::Replay>),
    /// Make the host pipelines of every pipeline the title ever created
    /// (runs with `VICTORY_PIPELINE_BOOK=1` write them down), so that the
    /// driver's cache on disk has them before a run needs one: once per
    /// change of the translator or the driver. Goes on where it was stopped.
    #[cfg(windows)]
    Warm {
        /// Processes at once (default: two thirds of the logical cores).
        #[arg(long)]
        jobs: Option<usize>,
        /// At most this many pipelines.
        #[arg(long)]
        limit: Option<usize>,
        /// Those made before too.
        #[arg(long)]
        again: bool,
        /// Only the pipelines earlier runs drew with.
        #[arg(long)]
        used: bool,
        /// (What the processes it starts are given.)
        #[arg(long, hide = true)]
        these: Option<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
    /// Check the host D3D12 device for mesh shaders and shader model 6.6.
    GpuCheck,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

/// Diagnostics: the heap bytes Rust code holds, for the statistics.
#[cfg(all(windows, feature = "diagnostics"))]
#[global_allocator]
static ALLOC: victory_gpu::CountingAlloc = victory_gpu::CountingAlloc;

fn main() -> Result<()> {
    victory_core::logging::init("info");
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
            effects,
        } => shader::run(paths, verbose || disasm, disasm, effects),
        Command::Translate {
            paths,
            jobs,
            pipelines,
            pipeline_selection,
            cache: true,
            ..
        } => {
            #[cfg(windows)]
            {
                translate::warm(paths, jobs.map(NonZeroUsize::get), pipelines, pipeline_selection)
            }
            #[cfg(not(windows))]
            {
                let _ = (paths, jobs, pipelines, pipeline_selection);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            cache: false,
            wave32,
            ..
        } => {
            #[cfg(windows)]
            {
                translate::run(paths, validate, pipelines, list, jobs.map(NonZeroUsize::get), wave32)
            }
            // Translation checks shaders with the Windows DXC host, which
            // other hosts do not have yet.
            #[cfg(not(windows))]
            {
                let _ = (paths, validate, pipelines, list, jobs, wave32);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            console,
            title_args: _,
        } => {
            let code = run::run(&exe, unimplemented, strict, execute, console)?;
            if code != 0 {
                // Preserve all 32 bits of the Windows process status.
                std::process::exit(code as i32);
            }
            Ok(())
        }
        #[cfg(windows)]
        Command::Replay(options) => replay::run(*options),
        #[cfg(windows)]
        Command::Warm { jobs, limit, again, used, these } => warm::run(jobs, limit, again, used, these),
        Command::Config => show_config(),
        Command::GpuCheck => gpu_check(),
    }
}

#[cfg(windows)]
fn gpu_check() -> Result<()> {
    let support = victory_gpu::host_mesh_support()
        .ok_or_else(|| anyhow::anyhow!("could not initialise VIctory's D3D12 host"))?;
    let status = |feature| match feature {
        Some(true) => "supported",
        Some(false) => "unsupported",
        None => "query failed",
    };
    println!("Mesh shaders: {}", status(support.mesh_shaders));
    println!("Shader model 6.6: {}", status(support.shader_model_6_6));
    anyhow::ensure!(
        support.supported(),
        "could not confirm both mesh pipeline requirements"
    );
    Ok(())
}

#[cfg(not(windows))]
fn gpu_check() -> Result<()> {
    anyhow::bail!(
        "gpu-check requires the Windows build of VIctory. On macOS run tools/macos/victory.sh gpu-check"
    )
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("fidelityfx  : {}", status(&cfg.fidelityfx_dir));
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    check_policy(&cfg)
}

/// Reports each configured path the asset policy refuses, and fails if
/// there is one, so that `victory config` exits nonzero.
fn check_policy(cfg: &victory_core::Config) -> Result<()> {
    let mut violations = 0;
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
            violations += 1;
        }
    }
    anyhow::ensure!(
        violations == 0,
        "{violations} configured path(s) violate the asset policy"
    );
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}

/// Writes a game-derived report (the `VICTORY_PREBUILD_KEYS` listing) to
/// `path`. Rule #1: the destination is checked first, so nothing is written
/// inside any VIctory source tree. Compiled where it is used (Windows) or
/// tested (any host).
#[cfg(any(windows, test))]
pub(crate) fn write_prebuild_keys(path: &Path, text: &str) -> Result<()> {
    victory_core::ensure_outside_repo(path)?;
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{Cli, check_policy, write_prebuild_keys};

    #[test]
    fn pipeline_selection_requires_cache_and_excludes_full_prewarm() {
        let selection = ["victory", "translate", "--pipeline-selection", "keys.txt"];
        assert!(Cli::try_parse_from(selection).is_err());
        let selected = selection.into_iter().chain(["--cache"]);
        assert!(Cli::try_parse_from(selected.clone()).is_ok());
        assert!(Cli::try_parse_from(selected.chain(["--pipelines"])).is_err());
        assert!(Cli::try_parse_from(["victory", "translate", "--cache", "--pipelines"]).is_ok());
    }

    fn outside() -> PathBuf {
        std::env::temp_dir().join("victory-config-test")
    }

    #[test]
    fn config_check_passes_paths_outside_the_repository() {
        let cfg = victory_core::Config {
            cache_dir: outside().join("cache"),
            data_dir: outside().join("data"),
            ..Default::default()
        };
        assert!(check_policy(&cfg).is_ok());
    }

    #[test]
    fn config_check_fails_a_path_inside_the_repository() {
        let cfg = victory_core::Config {
            game_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("game")),
            cache_dir: outside().join("cache"),
            data_dir: outside().join("data"),
            ..Default::default()
        };
        let err = check_policy(&cfg).unwrap_err();
        assert!(err.to_string().contains("1 configured path"), "{err}");
    }

    /// The `VICTORY_PREBUILD_KEYS` listing is game-derived (rule #1): a
    /// destination inside a source tree (marked with `.victory-root`) is
    /// refused and an existing file there is left byte for byte unchanged.
    #[test]
    fn prebuild_keys_refuse_repo_destinations() {
        let report = "0123456789abcdef\tGraphics(0)\tvertex, pixel\n";
        // A stand-in checkout: a temporary directory carrying the root marker,
        // with a sentinel file the refused write must not touch.
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join(victory_core::policy::ROOT_MARKER), "").unwrap();
        let sentinel = repo.path().join("keys.txt");
        let original = b"sentinel bytes, unchanged";
        std::fs::write(&sentinel, original).unwrap();
        let err = write_prebuild_keys(&sentinel, report).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref(),
                Some(victory_core::PolicyError::InsideRepo { .. })
            ),
            "{err}"
        );
        assert_eq!(std::fs::read(&sentinel).unwrap(), original);
        // An unmarked directory is written as asked.
        let out = tempfile::tempdir().unwrap();
        let dest = out.path().join("keys.txt");
        write_prebuild_keys(&dest, report).unwrap();
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), report);
    }
}
