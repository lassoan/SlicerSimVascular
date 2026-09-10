# Virtual Stent (SDFStent)

## Summary

Simulate stent deployment using SDFStent algorithm (provided by [svMorph](https://github.com/SimVascular/svMorph) Python package).

![](SDFStent01.jpg)

![](SDFStent02.jpg)

## Tutorial

- Go to `Sample Data` module and in `SimVascular` category click `Vessel01`. This loads a vessel segmentation and centerline. For using on your own data: segmentation can be created using `Segment Editor` module, centerline can be automatically created using `SlicerVMTK` extension's `Extract Centerline` module.
- Go to `Virtual Stent (SDFStent)` module
- Select `Vessel segment` -> `Vessel01 Segmentation`
- Select `Segment` -> `aorta`
- Select `Centerline curve` -> `Vessel01 Centerline`
- Select stent location by clicking the arrow button in the `Stent center point` row, then click on the stenosis (the narrowing where stent placement will be simulated)

![](SDFStentCenterPoint.jpg)

- Click the `Update` button to expand the vessel (deploy the stent). The generated expanded vessel model should be visible in about 10-20 seconds. First time it may take a few minutes to download require Python packages.
- Click the checkbox on the `Update` button to make the model update automatically whenever any parameter is changed.
- Increase and decrease the `Stent target radius` and other parameters. The generated mesh will be updated accordingly.

![](SDFStentAutoUpdate.jpg)

## Flared (funnel/trumpet shaped) stent

By default the stent radius is uniform along its length (svMorph's "pill" shape). One end of the stent can be flared into a funnel/trumpet shape:

- `Flared end`: which end of the stent is flared, relative to the direction of the input centerline. `Centerline start` flares the stent end that is closer to the first point of the centerline curve; `Centerline end` flares the opposite end. Select `None` for a uniform-radius stent.
- `Flare radius`: stent radius at the tip of the flared end (typically larger than the `Stent target radius`, but a smaller value can be used for a tapered stent).
- `Flare length`: length of the transition from the `Stent target radius` to the `Flare radius`, measured along the centerline from the flared stent end. The transition uses a smoothstep profile.

During deployment the whole radius profile is expanded proportionally, so when the stent body reaches the target radius, the flared end reaches the flare radius. The optional stent transform output warps the straight stent model accordingly, so a stent mesh transformed with it becomes flared as well.

Internally the stent surface is modeled by svMorph's variable-radius (tapered) capsule-chain signed distance field: one radius per stent axis vertex, interpolated along the centerline, so arbitrary radius profiles (not just flares) can be produced programmatically via `svmorph.core.stent_radius_profile` (from control points) or `svmorph.core.flared_stent_radius_profile`. All capsule end caps (at both ends of every segment, including the two stent ends) are deployed as flattened half ellipsoids instead of full spheres. This lets the radius profile express concave features — a wide capsule's spherical cap would otherwise bulge a full radius deep into a neighboring narrow region and wash the narrowing out — and rounds off a flared end smoothly without inflating a large spherical bulge around the vessel beyond the stent end. The cap height is adjustable in the Advanced section (`End cap height fraction`): the axial semi-axis of the caps as a fraction of the local stent radius, 0.35 by default; 1.0 reproduces svMorph's original spherical caps (pill shape). This requires an svmorph version with variable-radius stent support; the module installs a suitable version automatically.

For debugging and tweaking algorithm parameters, the Advanced section's `Output stent capsules model` shows this stent SDF geometry directly: the chain of tapered capsules is appended into a single model with a `CapsuleId` point scalar (the index of the capsule along the stent axis). The model is colored by capsule ID and shown semi-transparently by default, and updates live during deployment.

## References

[svMorph: Interactive geometry-editing tools for virtual patient-specific vascular anatomies](https://arxiv.org/abs/2210.07087)

```
@misc{pham2022svmorphinteractivegeometryeditingtools,
      title={svMorph: Interactive geometry-editing tools for virtual patient-specific vascular anatomies}, 
      author={Jonathan Pham and Sofia Wyetzner and Martin R. Pfaller and David W. Parker and Doug L. James and Alison L. Marsden},
      year={2022},
      eprint={2210.07087},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph},
      url={https://arxiv.org/abs/2210.07087}, 
}
```
