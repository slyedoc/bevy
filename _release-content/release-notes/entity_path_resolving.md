---
title: Feature name
authors: ["@Freyja-moth"]
pull_requests: [24018]
---

Paths can now be resolved to an entity and vice versa.
These use the `Name` component, following the `ChildOf` relationship.

```rust
fn character(world: &mut World, player: Entity) -> Result {
  let sword = world.get_entity_from_path("Items/Weapons/Sword", None)?;
    
  world.entity_mut(player)
    .add_one_related::<ItemOf>(sword);

  // Relative paths can also be resolved 
  let left_arm = world.get_entity_from_path("Arms/Left", Some(player))?; 
    
  // You can even use custom relationships.
  let apple = world.get_entity_from_relationship_path::<ItemOf>("Player/Satchel/Apple", None)?;

  Ok(())
}
```

With this a new variant has been added to `EntityTemplate` so that paths can be resolved in scenes

```rust
fn player() -> impl Scene {
  bsn! {
    #Player
    Player
    Wielding("Items/Weapons/Sword")
  }
}
```
