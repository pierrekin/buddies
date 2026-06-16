"""buddies: simulate -> process -> view.

    buddies simulate <sim> [run]            run the simulation
    buddies profile  <sim>                  time a truncated run's throughput
    buddies process  <sim> [run] [out]      prepare a run for viewing
    buddies view     <sim> [out]            open the viewer
    buddies show     <sim> [run] [out]      run whatever is missing, then view

Outputs live under output/<sim>/. A run name and an output name both default to
"default"; process's output name defaults to its source run, so one name can
carry through:

    buddies simulate mysim big --resolution 20
    buddies process  mysim big
    buddies view     mysim big

``show`` collapses that to one call, taking both stages' options and rerunning
only the stages whose options changed:

    buddies show mysim big --resolution 20 --decimate 2

Options (--resolution, --decimate, ...) follow the positionals.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

from buddies import profiling, simargs, store

ROOT = "output"
SIMS_DIR = "simulations"
SRC_DIR = os.path.join("src", "buddies")
DEFAULT_NAME = "default"
SIM_SIG_KEYS = ("resolution", "cfl", "capture_every", "max_steps")
# Keys we read off master.meta when building processed's source_sig. The
# CLI-arg keys plus the source-tree digest so any framework or sim-source
# edit propagates into processed's signature too.
MASTER_META_KEYS = (*SIM_SIG_KEYS, "src_digest")
# Files inside experiments/<sim>/ that don't produce artifacts; skipping
# them means tweaking a plot doesn't force a full simulate+process rerun.
NON_ARTIFACT_FILES = {"view.py"}


def sim_path(sim, run):
    return os.path.join(ROOT, sim, "sim", run)


def proc_path(sim, out):
    return os.path.join(ROOT, sim, "processed", out)


def _dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def available_sims():
    return [s for s in _dirs(SIMS_DIR) if os.path.exists(os.path.join(SIMS_DIR, s, "simulate.py"))]


def _die_listing(what, items, prefix=None):
    lines = [prefix] if prefix else []
    lines.append(f"available {what}:" if items else f"no {what} yet")
    lines += [f"  {i}" for i in items]
    sys.exit("\n".join(lines))


def _load_stage(sim, stage):
    """Import ``experiments/<sim>/<stage>.py`` by path, or None if absent."""
    path = os.path.join(SIMS_DIR, sim, f"{stage}.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(f"buddies_{sim}_{stage}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_simulate(sim):
    mod = _load_stage(sim, "simulate")
    if mod is None:
        sys.exit(f"no simulate.py for {sim!r} (looked in {SIMS_DIR}/{sim}/)")
    return mod


def _extra_views(sim):
    """The ``extra_views`` callable from the sim's optional ``view.py``, or
    None if it doesn't ship one (or doesn't export the function)."""
    mod = _load_stage(sim, "view")
    return getattr(mod, "extra_views", None) if mod is not None else None


def _processor(sim):
    """The sim's own ``process`` if it ships one, else the default processor."""
    mod = _load_stage(sim, "process")
    if mod is not None and hasattr(mod, "process"):
        return mod, mod.process
    from buddies import process as default_process

    return mod, default_process.process


def _src_digest(sim):
    """SHA-256 over the .py files that affect this sim's artifacts.

    Walks ``src/buddies/`` and ``experiments/<sim>/``, skipping caches and
    view-only files. The digest goes into the master's meta; ``show`` then
    treats any source-tree change as a reason to rerun simulate/process."""
    paths = []
    for root in (SRC_DIR, os.path.join(SIMS_DIR, sim)):
        for dp, _, files in os.walk(root):
            if "__pycache__" in dp.split(os.sep):
                continue
            for f in files:
                if not f.endswith(".py") or f in NON_ARTIFACT_FILES:
                    continue
                paths.append(os.path.join(dp, f))
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        h.update(b"\0")
        with open(p, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _sim_sig(args, sim):
    """The signature recorded for a master built with these args."""
    return {k: getattr(args, k) for k in SIM_SIG_KEYS} | {"src_digest": _src_digest(sim)}


def _read_meta(path):
    metafile = os.path.join(path, store.META)
    if not os.path.exists(metafile):
        return None
    with open(metafile) as f:
        return json.load(f)


def _stale(path, expected):
    """True if ``path`` has no meta or any ``expected`` key differs from it."""
    meta = _read_meta(path)
    if meta is None:
        return True
    return any(meta.get(k) != v for k, v in expected.items())


def _write_master(mod, sim, run, args):
    path = sim_path(sim, run)
    writer = store.Writer(path, {"kind": "master", "sim": sim, "run": run, **_sim_sig(args, sim)})
    mod.run(args, writer)
    return path


def _write_processed(proc, sim, run, out, pargs, source_sig):
    path = proc_path(sim, out)
    writer = store.Writer(
        path,
        {
            "kind": "processed",
            "sim": sim,
            "source_run": run,
            "out": out,
            "decimate": pargs.decimate,
            "percentile": pargs.percentile,
            "source": source_sig,
        },
    )
    proc(store.open_store(sim_path(sim, run)), pargs, writer)
    return path


def _proc_sig(pargs, run, source_sig):
    return {
        "decimate": pargs.decimate,
        "percentile": pargs.percentile,
        "source_run": run,
        "source": source_sig,
    }


def cmd_simulate(a):
    mod = _require_simulate(a.sim)
    ap = argparse.ArgumentParser(prog=f"buddies simulate {a.sim}", description=mod.__doc__)
    ap.add_argument("run", nargs="?", default=DEFAULT_NAME)
    simargs.add_sim_args(ap, **getattr(mod, "DEFAULTS", {}))
    ns = ap.parse_args(a.rest)
    args = simargs.sim_args(ns, mod.FREQ)
    print(f"wrote {_write_master(mod, a.sim, ns.run, args)}")


def cmd_profile(a):
    mod = _require_simulate(a.sim)
    ap = argparse.ArgumentParser(
        prog=f"buddies profile {a.sim}", description="truncated-run throughput profile"
    )
    ap.add_argument("--seconds", type=float, default=5.0, help="measurement window after warmup")
    ap.add_argument("--warmup", type=float, default=0.2, help="warmup before timing")
    ap.add_argument(
        "--breakdown", action="store_true",
        help="attribute time to step/capture/record (per-phase device sync; rate not comparable)",
    )
    simargs.add_sim_args(ap, **getattr(mod, "DEFAULTS", {}))
    ns = ap.parse_args(a.rest)
    args = simargs.sim_args(ns, mod.FREQ)

    prof = profiling.Profiler(args.xp, warmup_s=ns.warmup, budget_s=ns.seconds, breakdown=ns.breakdown)
    with profiling.session(prof):
        try:
            mod.run(args, profiling.ProfileWriter())
        except profiling.Stop:
            pass
    print(profiling.format_report(prof, args, a.sim))


def cmd_process(a):
    proc_mod, proc = _processor(a.sim)
    ap = argparse.ArgumentParser(prog=f"buddies process {a.sim}")
    ap.add_argument("run", nargs="?", default=DEFAULT_NAME, help="source run")
    ap.add_argument("out", nargs="?", default=None, help="output name (default: the run name)")
    simargs.add_process_args(ap, **(getattr(proc_mod, "DEFAULTS", {}) if proc_mod else {}))
    ns = ap.parse_args(a.rest)

    pargs = simargs.process_args(ns)
    out = ns.out or ns.run
    if not os.path.isdir(sim_path(a.sim, ns.run)):
        _die_listing(f"runs for {a.sim}", _dirs(os.path.join(ROOT, a.sim, "sim")), prefix=f"no run {ns.run!r}")
    master = store.open_store(sim_path(a.sim, ns.run))
    source_sig = {k: master.meta.get(k) for k in MASTER_META_KEYS}
    print(f"wrote {_write_processed(proc, a.sim, ns.run, out, pargs, source_sig)}")


def cmd_view(a):
    from buddies import viewer

    ap = argparse.ArgumentParser(prog=f"buddies view {a.sim}")
    ap.add_argument("out", nargs="?", default=DEFAULT_NAME)
    ap.add_argument("--fps", type=float, default=60.0, help="playback rate")
    ns = ap.parse_args(a.rest)

    if not os.path.isdir(proc_path(a.sim, ns.out)):
        _die_listing(f"outputs for {a.sim}", _dirs(os.path.join(ROOT, a.sim, "processed")), prefix=f"no output {ns.out!r}")
    st = store.open_store(proc_path(a.sim, ns.out))
    viewer.launch(
        st, title=f"{a.sim}/{ns.out}", fps=ns.fps,
        extra_views=_extra_views(a.sim),
    )


def cmd_show(a):
    from buddies import viewer

    sim_mod = _require_simulate(a.sim)
    proc_mod, proc = _processor(a.sim)

    ap = argparse.ArgumentParser(prog=f"buddies show {a.sim}")
    ap.add_argument("run", nargs="?", default=DEFAULT_NAME)
    ap.add_argument("out", nargs="?", default=None, help="output name (default: the run name)")
    simargs.add_sim_args(ap, **getattr(sim_mod, "DEFAULTS", {}))
    simargs.add_process_args(ap, **(getattr(proc_mod, "DEFAULTS", {}) if proc_mod else {}))
    ap.add_argument("--fps", type=float, default=60.0, help="playback rate")
    ns = ap.parse_args(a.rest)

    args = simargs.sim_args(ns, sim_mod.FREQ)
    pargs = simargs.process_args(ns)
    run = ns.run
    out = ns.out or ns.run
    sig = _sim_sig(args, a.sim)

    mpath = sim_path(a.sim, run)
    if _stale(mpath, sig):
        print(f"simulating {a.sim}/{run} ...")
        _write_master(sim_mod, a.sim, run, args)
    else:
        print(f"master {a.sim}/{run} up to date")

    ppath = proc_path(a.sim, out)
    if _stale(ppath, _proc_sig(pargs, run, sig)):
        print(f"processing {a.sim}/{out} ...")
        _write_processed(proc, a.sim, run, out, pargs, sig)
    else:
        print(f"processed {a.sim}/{out} up to date")

    viewer.launch(
        store.open_store(ppath), title=f"{a.sim}/{out}", fps=ns.fps,
        extra_views=_extra_views(a.sim),
    )


def main():
    ap = argparse.ArgumentParser(
        prog="buddies", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func, help_text in (
        ("simulate", cmd_simulate, "run the simulation"),
        ("profile", cmd_profile, "time a truncated run's throughput"),
        ("process", cmd_process, "prepare a run for viewing"),
        ("view", cmd_view, "open the viewer"),
        ("show", cmd_show, "run whatever is missing, then view"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("sim", nargs="?")
        sp.set_defaults(func=func)

    a, rest = ap.parse_known_args()
    if a.sim is None:
        _die_listing("simulations", available_sims())
    if a.sim not in available_sims():
        _die_listing("simulations", available_sims(), prefix=f"unknown simulation {a.sim!r}")
    a.rest = rest
    a.func(a)


if __name__ == "__main__":
    main()
