"""Absorbing boundaries, selectable by name so a simulation can swap one for
another without changing the rest of its setup.

``make`` returns the keyword arguments to hand to ``AcousticFDTD``:

- "sponge": an amplitude-damping layer (see ``edge_sponge``), folded into the
  step. Cheap, but reflects at grazing incidence.
- "pml": a convolutional PML that stays absorbing at all angles (coming).
"""

from buddies.sim import edge_sponge


def make(args, shape, dx):
    """The ``AcousticFDTD`` keyword args for the boundary ``args.boundary`` asks
    for, sized to ``shape`` cells at spacing ``dx`` (meters)."""
    if args.boundary == "sponge":
        return {"damping": edge_sponge(shape, dx, cells=args.sponge_cells)}
    if args.boundary == "pml":
        raise NotImplementedError("the PML boundary is not implemented yet")
    raise ValueError(f"unknown boundary {args.boundary!r}")
