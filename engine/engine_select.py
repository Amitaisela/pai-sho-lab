"""Resolves the "python"|"rust" engine choice to a PaiShoGame-shaped class.

See docs/superpowers/plans/2026-07-25-rust-engine-milestone-5-mutation-win-detection.md's
successor work and docs/superpowers/specs/2026-07-24-rust-engine-design.md §5.
Both classes are duck-typed drop-ins for each other (same methods/attributes,
no shared base class - see the design spec's "why duck-typing" note), so
callers can use whichever this returns without further branching.

The Rust engine is an optional build: `RustEngine` only exists once
`crates/pybind` has been built with maturin and the wheel installed (see
engine/RustEngine/crates/pybind/README or the Dockerfile). Requesting
"rust" without that raises a clear, actionable ImportError rather than the
default confusing one.
"""

DEFAULT_ENGINE = 'python'
ENGINE_CHOICES = ('python', 'rust')


def game_class(engine: str = DEFAULT_ENGINE):
    """The PaiShoGame-shaped class for `engine` ('python' or 'rust')."""
    if engine == 'python':
        from PythonEngine.PaiShoGame import PaiShoGame
        return PaiShoGame
    if engine == 'rust':
        try:
            from RustEngine import PaiShoGame
        except ImportError as e:
            raise ImportError(
                "Rust engine requested but not built. From engine/RustEngine/crates/pybind, "
                "run `maturin build --release` and `pip install` the wheel it writes to "
                "engine/RustEngine/target/wheels/ (or `maturin develop` inside a virtualenv)."
            ) from e
        return PaiShoGame
    raise ValueError(f"Unknown engine {engine!r}; expected one of {ENGINE_CHOICES}")


def engine_name_of(game) -> str:
    """Which engine produced a live game instance, inferred from its class.

    Used where only the instance is available (e.g. `serialize()`), not the
    original engine-choice string.
    """
    return 'rust' if type(game).__module__.split('.')[0] == 'RustEngine' else 'python'
