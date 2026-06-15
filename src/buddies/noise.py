"""Ambient noise sources for the FDTD.

Real acoustic environments are not silent. Thermal motion, distant
shipping, surface chop, and a thousand other contributions add up to a
broadband pressure floor everywhere in the medium. A line array running
in real water would hear it; an FDTD running in pure silence would not,
and would tell flattering lies about a comm system's robustness.

This module's ``AmbientNoise`` puts a fixed-position scatter of point
sources into the domain, each driven by an iid Gaussian volume-rate
vector at a configurable RMS. It is intentionally minimalist:

  * white (broadband) noise -- the FDTD's BPFs (e.g. speaker/mic
    transducers) shape it to whatever band the receiver actually sees;
  * spatially uncorrelated point sources -- the wave equation handles
    propagation, so a receiver hears the Green's-function-weighted sum
    of every source, which approaches a diffuse field as N grows;
  * fixed layout per instance -- the *spatial* noise field is identical
    across shots that vary only the drive RMS, so a noise sweep is a
    clean σ -> output mapping with no layout-randomisation jitter.

Bring your own model for coloured noise, anisotropy, or a single
localised interferer; this module's job is the white-noise common case
on which everything else can be built. The same Source contract every
other buddies source uses applies: ``sources()`` returns a list ready
to drop into ``AcousticFDTD(..., sources=[...])``."""

import numpy as np

from buddies.sim import Source


class AmbientNoise:
    """N point ambient noise sources at fixed positions in the FDTD
    domain interior, each driven by an iid Gaussian volume-rate vector.

    Layout is generated once per instance with ``layout_seed`` so the
    *same* spatial noise field can be re-used across shots that vary
    only the drive RMS -- a noise sweep is then a function of σ alone.
    ``drive_seed`` controls the noise realisation handed out by each
    ``sources()`` call: same seed + same shape = same realisation.
    """

    def __init__(self, n_sources, domain_size, margin=0.2, layout_seed=42):
        # ``margin`` must be larger than the FDTD's sponge depth, otherwise
        # noise sources land inside the sponge and the wave equation eats
        # most of their output before it ever reaches the receiver. The
        # default sponge in ``buddies.sim.edge_sponge`` is 15 cells deep
        # which is 150 mm at the typical dx=10 mm grid, so the default
        # ``margin=0.2`` clears it with room to spare. Pass a larger value
        # if you use a coarser grid or a deeper sponge.
        if n_sources < 0:
            raise ValueError(f"n_sources must be >= 0, got {n_sources}")
        if margin * 2 >= domain_size:
            raise ValueError(
                f"margin {margin} m leaves no room in a {domain_size} m domain"
            )
        rng = np.random.default_rng(layout_seed)
        xs = rng.uniform(margin, domain_size - margin, size=n_sources)
        ys = rng.uniform(margin, domain_size - margin, size=n_sources)
        self.positions = [(float(x), float(y)) for x, y in zip(xs, ys)]
        self.n_sources = n_sources
        self.domain_size = domain_size
        self.margin = margin
        self.layout_seed = layout_seed

    def sources(self, sigma, steps, dt, drive_seed=11):
        """Build per-shot ``Source`` instances driven by iid Gaussian
        noise. ``sigma`` is the per-source volume-rate RMS (m^2/s).

        ``sigma <= 0`` returns ``[]`` so a sweep can include a clean
        baseline shot without special-casing it at the call site."""
        if sigma <= 0 or self.n_sources == 0:
            return []
        rng = np.random.default_rng(drive_seed)
        noise = rng.normal(
            0.0, sigma, size=(self.n_sources, steps),
        ).astype(np.float32)

        def make_waveform(j):
            # Capture j and dt and the noise array; index by step number.
            def waveform(t):
                i = int(t / dt)
                if 0 <= i < steps:
                    return float(noise[j, i])
                return 0.0
            return waveform

        return [
            Source(pos=pos, waveform=make_waveform(j))
            for j, pos in enumerate(self.positions)
        ]
