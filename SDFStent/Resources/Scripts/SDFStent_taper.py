"""Variable-radius (tapered/flared) stent support for SDFStent.

svmorph models the deployed stent as a chain of capsules that all share a single
radius, so the deployed shape is always a "pill". This module generalizes the stent
to an arbitrary radius profile along the stent axis: each axis vertex has its own
radius, linearly interpolated along each capsule segment (a chain of tapered
capsules). This allows funnel/trumpet (flared) stent shapes, or any other radius
function of the position along the centerline.

install_tapered_sdf() replaces svmorph.core.deformation.smin_sdf_capsule_contact_sculpt
at run time. svmorph's compute_sdf_contact_displacements looks that function up through
its module globals on every call, so the replacement takes effect without editing the
installed svmorph package (which pip install/upgrade would overwrite). The replacement
accepts the original scalar radius as well; it differs from the original only in the
capsule end caps, which are flattened half ellipsoids instead of spheres (see
install_tapered_sdf).

Radius profiles are returned as dimensionless fractions of the nominal (body) stent
radius, so the same profile can be evaluated in any length unit and rescaled during
gradual deployment: the per-vertex radius at any point of the deployment is
fraction * current_nominal_radius.
"""

from __future__ import annotations

import numpy as np


def normalized_arc_positions(axis_points) -> np.ndarray:
    """Arc-length position of each stent axis vertex, normalized to [0, 1] from the first vertex."""
    points = np.asarray(axis_points, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = arc_positions[-1]
    if total_length <= 0.0:
        return np.zeros(len(points))
    return arc_positions / total_length


def flare_profile_fractions(axis_points, body_radius: float, flare_radius: float,
                            flare_length: float, flare_at_axis_start: bool) -> np.ndarray:
    """Per-vertex stent radius profile that flares from body_radius to flare_radius over
    flare_length at one end of the stent axis, with a smoothstep transition (trumpet/funnel
    shape). All lengths must be in the same unit. Returns radii as fractions of body_radius,
    one value per axis vertex."""
    points = np.asarray(axis_points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = arc_positions[-1]
    distance_from_flared_end = arc_positions if flare_at_axis_start else total_length - arc_positions
    t = np.clip(1.0 - distance_from_flared_end / float(flare_length), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    radii = float(body_radius) + (float(flare_radius) - float(body_radius)) * t
    return radii / float(body_radius)


def profile_fractions_from_control_points(axis_points, control_points) -> np.ndarray:
    """Per-vertex stent radius profile from arbitrary (normalized arc position, radius fraction)
    control points, linearly interpolated along the stent axis. control_points is an iterable of
    (position in [0, 1] measured from the first axis vertex, radius as a fraction of the nominal
    stent radius) pairs. Returns one radius fraction per axis vertex."""
    control_points = sorted((float(position), float(fraction)) for position, fraction in control_points)
    if not control_points:
        raise ValueError("At least one radius profile control point is required")
    positions = [position for position, _ in control_points]
    fractions = [fraction for _, fraction in control_points]
    return np.interp(normalized_arc_positions(axis_points), positions, fractions)


# Default axial semi-axis of the flattened half-ellipsoid capsule end caps, as a fraction of
# the local capsule radius. 1.0 reproduces the spherical caps of plain capsules. Flattened caps
# let the radius profile express concave features (and flared ends without a protruding ball):
# a wide capsule's spherical end cap would otherwise bulge a full radius deep into a
# neighboring narrow region, washing the narrowing out of the deployed shape.
FLATTENED_CAP_HEIGHT_FRACTION = 0.35


def install_tapered_sdf(cap_height_fraction: float = FLATTENED_CAP_HEIGHT_FRACTION) -> None:
    """Replace svmorph's uniform-radius capsule-chain SDF with a tapered variant that also
    accepts a per-vertex radius array (shape (V,), matching the stent axis vertices), linearly
    interpolated along each capsule segment.

    The end caps of every capsule (at both ends of each segment, including the two stent ends)
    are flattened into half ellipsoids: radial semi-axes = local stent radius, axial semi-axis
    = cap_height_fraction of it (1.0 reproduces plain spherical caps). Implemented by scaling
    up the axial overshoot in each segment's closest-point computation, so the caps flatten
    along their own segment direction. Compared to svmorph's spherical caps this lets the
    radius profile express concave features (a wide capsule's spherical cap would otherwise
    bulge into a neighboring narrow region) and rounds off a flared stent end smoothly instead
    of inflating a large ball around the vessel beyond the stent end.

    Safe to call multiple times; re-installs only when cap_height_fraction changes."""
    import jax as jx
    import jax.numpy as jnp
    from svmorph.core import deformation

    cap_height_fraction = float(cap_height_fraction)
    if not 0.0 < cap_height_fraction <= 1.0:
        raise ValueError("cap_height_fraction must be in (0, 1]")
    if getattr(deformation, "_sdfstent_tapered_install_key", None) == cap_height_fraction:
        return

    cap_axial_scale = 1.0 / cap_height_fraction

    @jx.jit
    def tapered_smin_sdf_capsule_contact_sculpt(rv, stent_vertices, r_current):
        # Same as svmorph.core.deformation.smin_sdf_capsule_contact_sculpt, except that
        # r_current may be a per-vertex radius array in addition to a scalar; the radius is
        # linearly interpolated along each capsule segment (radius at the closest axis point).
        r = jnp.broadcast_to(jnp.asarray(r_current), (stent_vertices.shape[0],))
        ba_all = jnp.diff(stent_vertices, axis=0)
        pa_all = rv[:, None, :] - stent_vertices[None, :-1, :]
        ba_dot_pa_all = jnp.sum(pa_all * ba_all[None, :, :], axis=-1)
        ba_dot_ba_all = jnp.sum(ba_all**2, axis=-1)
        h_unclamped_all = ba_dot_pa_all / ba_dot_ba_all
        h_all = jnp.clip(h_unclamped_all, 0, 1)
        # Flattened end caps: the axial overshoot beyond the segment ends (nonzero only where
        # the closest-point parameter clamps) is scaled up, which turns each spherical cap into
        # a half ellipsoid with axial semi-axis cap_height_fraction * radius.
        axis_to_point_all = (pa_all - h_all[:, :, None] * ba_all[None, :, :]
                             + ((h_unclamped_all - h_all) * (cap_axial_scale - 1.0))[:, :, None] * ba_all[None, :, :])
        dist_all = jnp.linalg.norm(axis_to_point_all, axis=-1)[..., None]
        direction_all = axis_to_point_all / dist_all
        dist_all_squeezed = jnp.squeeze(dist_all, axis=-1)  # shape: (num_mesh_points, num_segments)
        r_at_closest_point_all = r[None, :-1] + h_all * (r[1:] - r[:-1])[None, :]
        dist_to_surface_all = dist_all_squeezed - r_at_closest_point_all
        final_dist_to_surface, final_direction = jx.vmap(deformation.compute_min_dist_and_direction)(
            dist_to_surface_all, direction_all)
        final_dist_to_surface = final_dist_to_surface[:, None]
        return final_dist_to_surface, final_direction

    deformation.smin_sdf_capsule_contact_sculpt = tapered_smin_sdf_capsule_contact_sculpt
    deformation._sdfstent_tapered_install_key = cap_height_fraction
