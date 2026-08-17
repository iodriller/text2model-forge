# Text2Model Forge review rubric

This rubric separates deterministic checks from decisions that require a human
to inspect the asset. It is deliberately independent of genre, species, game,
and visual style. A project can add stricter requirements in its own profile;
Text2Model Forge must not silently impose one project's taste on another.

## Deterministic gates

The pipeline records these as metrics or hard failures where the relevant
stage can measure them:

- requested files exist, parse, and match their recorded SHA-256 digests;
- image dimensions, alpha masks, frame counts, and atlas bounds are valid;
- mesh topology and component metrics are captured without treating any one
  proxy (for example raw component count) as visual quality;
- required named components, equipment, states, and animations are present;
- handedness and attachment contracts agree with the compiled D0 spec;
- lineage contains every upstream producer and its license status;
- release packaging fails closed on missing approval or uncleared lineage.

Passing these checks means the artifact is structurally reviewable. It does
not mean that it looks good or is suitable for a particular product.

## Human gates

At each applicable review stage, assess only the requirements in the compiled
asset spec and the consuming project's documented art direction:

1. **Identity and intent** — the result is recognizably the described asset,
   and repeated views or frames show the same design.
2. **Silhouette and scale** — the important forms remain legible at the target
   camera distance and output size.
3. **Parts and attachment** — required parts occur exactly as specified and
   stay connected to the correct parent, side, or joint.
4. **Motion or state change** — requested actions and articulated states have
   a readable beginning, change, and end; static assets are not judged here.
5. **Surface coherence** — materials, texture scale, lighting assumptions, and
   palette are internally consistent and follow the run's creative direction.
6. **Technical fitness** — topology, deformation, UVs, pivots, collision, and
   runtime behavior are appropriate for the declared target, not merely valid
   in isolation.
7. **Evidence quality** — review views expose the features under judgment.
   Hidden, cropped, or ambiguous evidence is grounds for retry, not approval.
8. **Originality and lineage** — the reviewer sees no obvious copied identity,
   trademarked insignia, or unexplained source material, and all source terms
   are recorded before release.

An approval is evidence-bound and specific to one stage. It is not a blanket
claim that the asset is legally clear, aesthetically strong, or production
ready for every target.

## Project-specific extensions

Keep title-specific art direction, named characters, target-engine budgets,
and owner feedback in the consuming project's repository. Express measurable
requirements through typed profiles or worker requests. Do not edit this
general rubric to encode one asset's exception.

When a rejection reveals a reusable failure mode, add a generic regression
test or deterministic validator if possible. Otherwise record the correction
in the run's append-only human decision history; do not turn subjective taste
into an undocumented global default.
