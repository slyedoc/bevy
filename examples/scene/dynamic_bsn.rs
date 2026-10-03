//! Demonstrates loading scenes from `.bsn` asset files.
//!
//! A `.bsn` file is data, not code: it is parsed at runtime and resolved against the type registry,
//! so it can be edited — and reloaded — without recompiling. This example shows the pieces that
//! makes that useful:
//!
//! 1. **Loading a `.bsn` file as an asset** and spawning it with [`ScenePatchInstance`]
//!    (`assets/scenes/dynamic_bsn_example.bsn`).
//! 2. **One `.bsn` file inheriting another** with `:"path.bsn"`, overriding individual fields —
//!    the styled `#Confirm` button inside that same file.
//! 3. **Inheriting a `.bsn` file from Rust** inside a [`bsn!`] macro, again with `:"path.bsn"`.
//! 4. **`#Name`s**, which become [`Name`] components and are how Rust code finds a particular
//!    entity inside a spawned scene.
//! 5. **Enum patching**, which is match-or-reset: naming a different variant replaces the whole
//!    value rather than merging into it.
//!
//! Note the difference in what the two forms accept: the `bsn!` macro below uses Rust function
//! calls (`px(24)`, `Color::srgb(...)`) and could use constants like `Color::WHITE`, while a `.bsn`
//! file supports neither, and spells the same values out as struct literals
//! (`Color::Srgba(Srgba { red: 0.95, ... })`).
//!
//! ## Hot reload
//!
//! Run the example with the `file_watcher` feature:
//!
//! ```sh
//! cargo run --example dynamic_bsn --features file_watcher
//! ```
//!
//! then edit `assets/scenes/dynamic_bsn_example.bsn` or `assets/scenes/dynamic_bsn_button.bsn`
//! while it is running — change a color, a size, or the button's label — and save. The changed
//! file is reloaded, every scene that inherits from it is re-resolved, and the spawned entities are
//! rebuilt in place. Editing the *base* button file updates both buttons at once.
//!
//! Requires the `bsn_asset` cargo feature (enabled by default).

use bevy::prelude::*;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins).add_systems(Startup, setup);

    // Bevy's CI runs this example headlessly and screenshots it; the system below turns "the scene
    // silently never spawned" and "the scene spawned with the wrong values" into a failed run.
    // This is testing infrastructure, not something you need in your own app.
    #[cfg(feature = "bevy_ci_testing")]
    app.add_systems(Update, verify_spawned_scenes);

    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);

    // 1. Spawn the `.bsn` file directly. The entity gets its components as soon as the asset and
    //    all of its dependencies (here: the logo image, and the button scene the `#Confirm` child
    //    inherits from) have loaded.
    commands.spawn(ScenePatchInstance(
        asset_server.load("scenes/dynamic_bsn_example.bsn"),
    ));

    // 2. Inherit from the same button scene in Rust, overriding where it sits and how it looks.
    //    `queue_spawn_scene` waits until "scenes/dynamic_bsn_button.bsn" is loaded. Fields the
    //    macro does not mention keep the values the asset gave them, so this button is still
    //    56 logical pixels tall and still centers its contents.
    //
    //    Children of a base and of the scene inheriting from it are appended rather than merged
    //    pairwise, which is why the button scene ships without a label: each user supplies its own.
    commands.queue_spawn_scene(bsn! {
        :"scenes/dynamic_bsn_button.bsn"
        #OverrideButton
        Node {
            position_type: PositionType::Absolute,
            bottom: px(24),
            right: px(24),
        }
        BackgroundColor(Color::srgb(0.15, 0.35, 0.15))
        Children [ #OverrideLabel Text("Overridden!") ]
    });
}

/// Fails the run if either spawn path has not produced the expected entities within
/// `DEADLINE_FRAMES` frames.
///
/// Scenes spawn asynchronously — the `.bsn` files and the logo image all have to load first — so
/// this runs every frame until it succeeds, and only then stops. Anything still wrong at the
/// deadline panics with the name of the check that failed.
#[cfg(feature = "bevy_ci_testing")]
fn verify_spawned_scenes(
    names: Query<(Entity, &Name)>,
    nodes: Query<&Node>,
    texts: Query<&Text>,
    backgrounds: Query<&BackgroundColor>,
    image_nodes: Query<&ImageNode>,
    children: Query<&Children>,
    images: Res<Assets<Image>>,
    frames: Res<bevy::diagnostic::FrameCount>,
    mut verified: Local<bool>,
) {
    /// Frames to give both scenes before declaring the run a failure.
    const DEADLINE_FRAMES: u32 = 200;

    if *verified {
        return;
    }

    /// Returns early with a description of what did not hold.
    macro_rules! check {
        ($condition:expr, $($message:tt)*) => {
            if !$condition {
                let failure = format!($($message)*);
                assert!(
                    frames.0 < DEADLINE_FRAMES,
                    "dynamic BSN example never reached its expected state: {failure}",
                );
                return;
            }
        };
    }

    /// Looks up the single entity carrying a given `#Name`, or reports it as missing.
    macro_rules! by_name {
        ($name:literal) => {{
            let found = names
                .iter()
                .find(|(_, name)| name.as_str() == $name)
                .map(|(entity, _)| entity);
            check!(
                found.is_some(),
                concat!("no entity named `", $name, "` yet")
            );
            found.unwrap()
        }};
    }

    // --- Path 1: the scene loaded straight from `dynamic_bsn_example.bsn`. ---

    let root = by_name!("Root");
    let root_node = nodes.get(root).expect("the root has no `Node`");
    check!(
        root_node.flex_direction == FlexDirection::Column,
        "root's flex_direction is {:?}, expected Column",
        root_node.flex_direction,
    );

    // `#Logo`'s `ImageNode` names an asset path, which makes the image a dependency of the scene:
    // reaching this point means the handle resolved and the image finished loading.
    let logo = by_name!("Logo");
    let logo_image = image_nodes.get(logo).expect("the logo has no `ImageNode`");
    check!(
        images.contains(&logo_image.image),
        "the logo's image has not loaded",
    );

    let title = by_name!("Title");
    check!(texts.contains(title), "the title has no `Text` component");

    // `#Confirm` is the `.bsn`-inherits-`.bsn` button: a `Val` variant switch over the base's
    // `Val::Px(180.0)`, plus its own background color and label.
    let confirm = by_name!("Confirm");
    let confirm_node = nodes
        .get(confirm)
        .expect("the confirm button has no `Node`");
    check!(
        confirm_node.width == Val::Percent(30.0),
        "the confirm button's width is {:?}, expected Percent(30.0)",
        confirm_node.width,
    );
    check!(
        confirm_node.height == Val::Px(56.0),
        "the confirm button did not inherit the base scene's height, it is {:?}",
        confirm_node.height,
    );
    check!(
        is_srgb(background(&backgrounds, confirm), 0.1, 0.3, 0.55),
        "the confirm button's background color was not overridden",
    );
    check!(
        children.get(confirm).is_ok_and(|kids| kids.len() == 1),
        "the confirm button should have exactly one child, its label",
    );

    // --- Path 2: the `bsn!` macro inheriting the same base scene from Rust. ---

    let override_button = by_name!("OverrideButton");
    let override_node = nodes
        .get(override_button)
        .expect("the overridden button has no `Node`");
    check!(
        override_node.position_type == PositionType::Absolute,
        "the overridden button is not absolutely positioned",
    );
    check!(
        override_node.bottom == Val::Px(24.0) && override_node.right == Val::Px(24.0),
        "the overridden button is not offset from the bottom right corner",
    );
    check!(
        override_node.height == Val::Px(56.0),
        "the overridden button did not inherit the base scene's height, it is {:?}",
        override_node.height,
    );
    check!(
        is_srgb(background(&backgrounds, override_button), 0.15, 0.35, 0.15),
        "the overridden button's background color was not overridden",
    );

    let override_label = by_name!("OverrideLabel");
    check!(
        texts
            .get(override_label)
            .is_ok_and(|text| text.0 == "Overridden!"),
        "the overridden button's label text is wrong",
    );

    info!("dynamic BSN example verified on frame {}", frames.0);
    *verified = true;
}

/// The background color of `entity`, which every button in this example has.
#[cfg(feature = "bevy_ci_testing")]
fn background(backgrounds: &Query<&BackgroundColor>, entity: Entity) -> Color {
    backgrounds
        .get(entity)
        .expect("entity has no `BackgroundColor`")
        .0
}

/// Whether `color` is the opaque sRGB color `(red, green, blue)`, allowing for rounding.
#[cfg(feature = "bevy_ci_testing")]
fn is_srgb(color: Color, red: f32, green: f32, blue: f32) -> bool {
    let actual = Srgba::from(color);
    (actual.red - red).abs() < 1e-5
        && (actual.green - green).abs() < 1e-5
        && (actual.blue - blue).abs() < 1e-5
        && (actual.alpha - 1.0).abs() < 1e-5
}
