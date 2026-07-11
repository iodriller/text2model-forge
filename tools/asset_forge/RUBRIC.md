# Asset Forge art-direction rubric

The critic module (`assetforge critique`) enforces the measurable half of this rubric on
every build. The judgment half below is applied by the art-directing agent to every
acceptance board BEFORE the owner sees it; a board that fails any line here goes back
into iteration (adjust master geometry, prompt, seed, bake parameters), not to review.

## Measurable gates (enforced in code — `assetforge/critic.py`)

- Masked mean brightness within the dark-fantasy band; saturation muted (no neon, no pastel).
- Forms readable at 96 px gameplay height (edge-energy floor).
- One palette across consecutive frames (hue coherence — no identity flicker).
- Every sheet shares the idle/south palette (one character everywhere).
- Plus mechanical QA: 4 directions, shared baseline, real motion, no clipped frames.

## Judgment gates (applied by the reviewing agent on the acceptance board)

1. **Identity**: same character in every frame of every action — same armor, same
   emblem, same silhouette weight. Any frame that could be a different unit fails.
2. **Equipment**: exactly one sword and one shield (footman), always attached to the
   correct hand/arm, never missing, never duplicated, never floating.
3. **Attack reads as an attack**: visible anticipation (weapon back/up), a strike frame
   with clear extension toward the enemy, recovery. At 96 px a player must be able to
   say "it swung".
4. **Death reads as a collapse**: the body ends on the ground, inside frame, in every
   direction. Standing deaths and clipped corpses fail.
5. **Tone**: dark high fantasy, battle-worn, Warcraft-3-adjacent — not toy soldier, not
   Minecraft/voxel, not sticker-flat, not photoreal.
6. **Small-size read**: at gameplay size the head/torso/weapon grouping must remain
   distinct; if it becomes one blob, fail regardless of how good the 768 px frame looks.
7. **Creature anatomy precedes paint**: a Goblin must read as a Goblin in an unpainted
   front face, side face, and pure silhouette. Green human heads, human knight posture,
   and surface-only tusks fail. Ogres, dragons, and other families require their own
   morphology contract; never force them through a humanoid repaint.
8. **Review framing exposes anatomy**: creature acceptance boards require large front
   and side face crops, a pure silhouette, posture, attack phases, and 96 px gameplay
   scale. A whole-body-only board cannot be approved.

## Owner feedback log (append-only; every rejection becomes a permanent rule)

- 2026-07: Primitive block figures rejected — "toy soldier / Minecraft" is an
  automatic fail no matter what QA says.
- 2026-07: Sword/shield missing in some frames rejected — equipment presence is
  non-negotiable in every frame.
- 2026-07: Attack that barely moves or "rolls sideways" rejected — motion must be a
  professional strike arc.
- 2026-07: Bright/pale palettes rejected — direction is "high fantasy but dark a bit".
- 2026-07-10: White-painted helmet reads as bare/white head — helmets must read dark
  gunmetal steel. Hair clipping through the helmet dome is a geometry fail.
- 2026-07-10: Death sprawl must fit the cell in east/west — framing wide enough for a
  full-body collapse is part of the death acceptance, not an afterthought.
- 2026-07-11: Green paint on a human Warrior face and upright knight posture rejected.
  Species-defining skull, brow, snout, jaw, tusks, ears, body proportions, and posture
  must be authored geometry shared by every animation.
