use bevy_ecs::{bundle::Bundle, component::Component, reflect::ReflectComponent};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_scene::{bsn, Scene};

#[cfg(feature = "render_materials")]
pub(crate) use material::*;

/// Marker that tells us we want to fill in the `MaterialNode` with the alpha material.
///
/// Without the `render_materials` feature this is an inert marker: the checkerboard is not drawn,
/// and a custom render backend can paint it by joining this entity's `ComputedNode`.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Default)]
pub struct AlphaPattern;

/// Scene fragment which makes the entity it is merged into display the alpha checkerboard behind
/// its (possibly translucent) background.
#[cfg(feature = "render_materials")]
pub(crate) fn alpha_pattern() -> impl Scene {
    use bevy_ui_render::ui_material::MaterialNode;

    bsn! {
        AlphaPattern
        MaterialNode::<AlphaPatternMaterial>
    }
}

/// Scene fragment which would display the alpha checkerboard, but the `render_materials` feature
/// is off, so it only leaves the [`AlphaPattern`] marker behind.
#[cfg(not(feature = "render_materials"))]
pub(crate) fn alpha_pattern() -> impl Scene {
    bsn! {
        AlphaPattern
    }
}

/// Bundle form of [`alpha_pattern`], for the deprecated `*_bundle` template functions.
#[cfg(feature = "render_materials")]
pub(crate) fn alpha_pattern_bundle() -> impl Bundle {
    use bevy_ui_render::ui_material::MaterialNode;

    (
        AlphaPattern,
        MaterialNode::<AlphaPatternMaterial>(bevy_asset::Handle::default()),
    )
}

/// Bundle form of [`alpha_pattern`], for the deprecated `*_bundle` template functions.
#[cfg(not(feature = "render_materials"))]
pub(crate) fn alpha_pattern_bundle() -> impl Bundle {
    AlphaPattern
}

#[cfg(feature = "render_materials")]
mod material {
    use super::AlphaPattern;
    use bevy_app::Plugin;
    use bevy_asset::{Asset, Assets, Handle};
    use bevy_ecs::{
        lifecycle::Add,
        observer::On,
        resource::Resource,
        system::{Query, Res},
        world::FromWorld,
    };
    use bevy_reflect::TypePath;
    use bevy_render::render_resource::AsBindGroup;
    use bevy_shader::ShaderRef;
    use bevy_ui_render::ui_material::{MaterialNode, UiMaterial};

    #[derive(AsBindGroup, Asset, TypePath, Default, Debug, Clone)]
    pub(crate) struct AlphaPatternMaterial {}

    impl UiMaterial for AlphaPatternMaterial {
        fn fragment_shader() -> ShaderRef {
            "embedded://bevy_feathers/assets/shaders/alpha_pattern.wesl".into()
        }
    }

    #[derive(Resource)]
    pub(crate) struct AlphaPatternResource(pub(crate) Handle<AlphaPatternMaterial>);

    impl FromWorld for AlphaPatternResource {
        fn from_world(world: &mut bevy_ecs::world::World) -> Self {
            let mut ui_materials = world
                .get_resource_mut::<Assets<AlphaPatternMaterial>>()
                .unwrap();
            Self(ui_materials.add(AlphaPatternMaterial::default()))
        }
    }

    /// Observer to fill in the material handle (since we don't have access to the materials asset
    /// in the template)
    fn on_add_alpha_pattern(
        add: On<Add<AlphaPattern>>,
        mut q_material_node: Query<&mut MaterialNode<AlphaPatternMaterial>>,
        r_material: Res<AlphaPatternResource>,
    ) {
        if let Ok(mut material) = q_material_node.get_mut(add.entity) {
            material.0 = r_material.0.clone();
        }
    }

    /// Plugin which registers the systems for updating the button styles.
    pub struct AlphaPatternPlugin;

    impl Plugin for AlphaPatternPlugin {
        fn build(&self, app: &mut bevy_app::App) {
            app.add_observer(on_add_alpha_pattern);
        }
    }
}
