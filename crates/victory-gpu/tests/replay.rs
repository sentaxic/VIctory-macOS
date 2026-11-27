//! The frame recorder and replayer on this machine's own GPU (skipped
//! without one): a made-up frame (`victory_gpu::replay::testing`) runs
//! through the host layer with the recorder on, as a title's frames do, one
//! frame of it is captured, and the capture is run again without the host
//! layer, in every way a replay can be sent. Every run must leave the
//! picture the frame presented, and the buffer its passes wrote.
//!
//! One process, one device, one recorder: everything in one test.
#![cfg(windows)]

use victory_gpu::replay::play::{self, Detail, Options, Player, Waits};
use victory_gpu::replay::record;
use victory_gpu::replay::testing::{self, RESULTS, SIZE, TARGET};

#[test]
fn a_frame_recorded_through_the_host_layer_is_replayed() {
    let dir = tempfile::tempdir().unwrap();
    // Asked for as a title's run asks: by the environment, before the host
    // device is made; five frames after the world's first.
    // SAFETY: no other thread of this process reads the environment yet.
    unsafe {
        std::env::set_var("VICTORY_RECORD", dir.path());
        std::env::set_var("VICTORY_RECORD_AT", "w5");
    }
    let Some(made) = testing::run(14, 4) else {
        eprintln!("skipped: no host GPU");
        return;
    };
    assert!(record::on(), "the host layer started the recorder");
    let capture = record::finished().expect("a frame was captured");
    assert_eq!(capture, dir.path().join("capture-1"));
    let notes = std::fs::read_to_string(capture.join("notes.txt")).unwrap();
    println!("{notes}");
    // The frame asked for, whole, with nothing left out.
    assert!(notes.contains("frame 9 (world frame 5)"), "{notes}");
    assert!(notes.contains("3 submissions, 4 lists"), "{notes}");
    for bad in ["not recorded", "not in the capture", "not whole", "lost its hooks", "not seen made"] {
        assert!(!notes.contains(bad), "{bad}: {notes}");
    }
    assert!(std::fs::read(capture.join("frame-0.png")).unwrap().starts_with(b"\x89PNG"));

    // As the runtime runs it: each submission waited for.
    let options = Options {
        loops: 3,
        ..Default::default()
    };
    let mut player = Player::load(dir.path(), &options).unwrap();
    let frame = player.frame();
    assert_eq!((frame.number, frame.world, frame.partial), (9, 4, false));
    assert_eq!(frame.submits.len(), 3);
    assert_eq!(frame.submits.iter().map(|s| s.lists.len()).collect::<Vec<_>>(), [1, 2, 1]);
    // The view made before the second submission and the one replaced
    // before the third, and what the CPU wrote before each.
    assert_eq!(frame.submits.iter().map(|s| s.views.len()).collect::<Vec<_>>(), [0, 1, 1]);
    assert!(frame.submits.iter().all(|s| !s.deltas.is_empty()));
    let report = player.run(&options).unwrap();
    println!("{}", report.text("each waited for"));
    assert_eq!(report.picture, Some((f64::INFINITY, true)), "the picture presented");
    assert_eq!(report.frame_ms.len(), 3);
    assert_eq!(report.submits.len(), 3);
    assert!(report.submits.iter().all(|s| s.1.is_some_and(|ms| ms > 0.0)));
    let picture = |player: &mut Player| {
        let rows = player.contents(player.named(TARGET).unwrap()).unwrap().unwrap();
        // The copyable layout of a texture this wide has no padding.
        assert_eq!(rows.len(), (SIZE * SIZE * 4) as usize);
        rows
    };
    let results = |player: &mut Player| {
        let bytes = player.contents(player.named(RESULTS).unwrap()).unwrap().unwrap();
        bytes[..made.results.len()].to_vec()
    };
    assert!(picture(&mut player) == made.picture, "the picture the frame copied out");
    assert_eq!(results(&mut player), made.results);
    // The picture is one of things drawn: not one colour.
    let texels: std::collections::HashSet<&[u8]> = made.picture.chunks(4).collect();
    assert!(texels.len() > 100, "{} colours", texels.len());

    // The textures no command names hold what they were given: every
    // format and shape was taken out of the GPU's memory and put back.
    assert!(made.others.len() >= 20, "{} textures of other formats", made.others.len());
    let mut wrong = Vec::new();
    for other in &made.others {
        let id = player.named(&other.name).unwrap_or_else(|| panic!("{} is in the capture", other.name));
        assert!(!other.subs.is_empty(), "{}", other.name);
        for (sub, rows) in other.subs.iter().enumerate() {
            let now = player.contents_of(id, sub as u32).unwrap().unwrap();
            if !rows.are_in(&now) {
                // The first texels that differ, as they were given and as
                // they are.
                let differ: Vec<String> = (0..rows.rows)
                    .flat_map(|y| (0..rows.row / 4).map(move |x| (y, x)))
                    .filter_map(|(y, x)| {
                        let was = rows.bytes.get(y * rows.row + x * 4..y * rows.row + x * 4 + 4)?;
                        let is = now.get(y * rows.pitch + x * 4..y * rows.pitch + x * 4 + 4)?;
                        (was != is).then(|| format!("({x},{y}) {was:02x?} -> {is:02x?}"))
                    })
                    .take(6)
                    .collect();
                wrong.push(format!("{} subresource {sub}: {}", other.name, differ.join(", ")));
            }
            // (What a depth target read back when it was made is not
            // nothing.)
            assert!(rows.bytes.iter().any(|&b| b != 0), "{} subresource {sub} is all zero", other.name);
        }
    }
    assert!(wrong.is_empty(), "{} subresources differ:\n{}", wrong.len(), wrong.join("\n"));

    // Sent without waits, as one submission, in pieces, with a timestamp
    // after every command, with an idle gap: the same picture.
    for (what, o) in [
        ("no waits", Options { waits: Waits::None, loops: 2, ..Default::default() }),
        ("one submission", Options { waits: Waits::OneSubmit, loops: 2, ..Default::default() }),
        ("in pieces", Options { detail: Detail::Pieces, loops: 2, ..Default::default() }),
        ("stamps", Options { detail: Detail::Stamps, loops: 2, ..Default::default() }),
        ("paced", Options { pace: 2.0, loops: 2, ..Default::default() }),
        // Without the host layer's two clamps, their calls reading the
        // arguments kept from a run with them; and the frame after it.
        ("without clamps", Options { keep_clamps: true, loops: 2, ..Default::default() }),
        ("after it", Options { loops: 2, ..Default::default() }),
    ] {
        let report = player.run(&o).unwrap();
        println!("{}", report.text(what));
        assert_eq!(report.picture, Some((f64::INFINITY, true)), "{what}");
        assert!(picture(&mut player) == made.picture, "{what}");
        assert_eq!(results(&mut player), made.results, "{what}");
        if o.keep_clamps {
            assert!(
                report.notes.iter().any(|n| n.starts_with("2 argument clamps were left out")),
                "{:?}",
                report.notes
            );
        }
        if o.detail != Detail::Off {
            let kinds: Vec<&str> = report.categories.iter().map(|c| c.0.as_str()).collect();
            for kind in ["draws", "indirect draws", "dispatches", "clears"] {
                assert!(kinds.contains(&kind), "{what}: {kinds:?}");
            }
            // One plain and one with indices; the third by its arguments.
            let draws = report.categories.iter().find(|c| c.0 == "draws").unwrap();
            assert_eq!(draws.2, 2, "{what}");
            assert!(report.shaders.iter().any(|s| s.2.contains("replay test: draw")), "{what}: {:?}", report.shaders);
            if o.detail == Detail::Pieces {
                // The frame's own time is shared out among the pieces: the
                // shares of a loop add up to its submissions' times.
                let shared: f64 = report.categories.iter().map(|c| c.3.unwrap()).sum();
                let (least, most) = report.frame_ms.iter().fold((f64::MAX, 0f64), |m, &f| (m.0.min(f), m.1.max(f)));
                assert!(shared > least * 0.5 && shared < most * 1.5, "{shared} of {least}..{most}");
            }
        }
    }

    // Without the draws' pipeline: the cleared target alone, and the report
    // says what was left out.
    let skipped = Options {
        skip_shaders: vec!["replay test: draw".into()],
        loops: 1,
        ..Default::default()
    };
    let report = player.run(&skipped).unwrap();
    println!("{}", report.text("without the draws"));
    assert!(report.picture.is_some_and(|p| !p.1));
    assert!(report.notes.iter().any(|n| n == "--skip-shader left out 5 commands"), "{:?}", report.notes);
    let cleared = picture(&mut player);
    assert!(cleared.chunks(4).all(|t| t == &cleared[..4]), "one colour");
    assert_eq!(results(&mut player), made.results, "the passes still ran");

    // Both, turn by turn; then the frame as it was, again.
    drop(player);
    let compared = play::replay(
        dir.path(),
        &Options {
            compare: true,
            skip_shaders: vec!["replay test: draw".into()],
            loops: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(compared.len(), 2);
    assert_eq!(compared[0].1.picture, Some((f64::INFINITY, true)));
    assert!(compared[1].1.picture.is_some_and(|p| !p.1));

    // The capture's own shaders, taken from a folder, give the same frame;
    // a folder without them changes nothing either.
    let listing = {
        let mut lines = Vec::new();
        play::listing(dir.path(), (0, 99), &mut |line| lines.push(line.to_owned())).unwrap();
        lines
    };
    assert!(listing.iter().any(|l| l.contains("replay test: draw")), "pipelines are named");
    assert!(listing.iter().any(|l| l.starts_with("list 3 of submission 2")));
    let shaders = dir.path().join("empty");
    std::fs::create_dir(&shaders).unwrap();
    let with = play::replay(
        dir.path(),
        &Options {
            shaders: Some(shaders),
            loops: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(with[0].1.picture, Some((f64::INFINITY, true)));
    assert!(with[0].1.notes.iter().any(|n| n.starts_with("0 of 3 pipelines made with shaders of")), "{:?}", with[0].1.notes);
}
