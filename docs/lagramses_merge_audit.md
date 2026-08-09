# lagRamses sink-merger audit

Audited source: `/home/kjhan/BACKUP/lagRamses`, branch `fdm-dev`, commit
`10b9db4290bc4e5250b95ecdf00ba9e9b4a5e7ab` on 2026-08-09.

## Active source resolution

`bin/Makefile` sets `PATCH=../patch/lagRamses`, but its `VPATH` also searches
`../patch/cuRamses`. The object list requests `sink_particle.kjhan.o`, so the
implementation inspected here is:

```text
patch/cuRamses/sink_particle.kjhan.f90
patch/cuRamses/kjhan.f90
```

`create_sink` calls `merge_sink(1)` once per coarse step, after sink positions
and velocities are updated and before new sink clouds are created.

## Exact numerical gate

`merge_sink` calculates

```fortran
dx_min = (boxlen / nx_loc) * 2**(-nlevelmax) / aexp
```

and links sinks when their periodic minimum-image distance satisfies

```fortran
rr <= rmerge**2 * dx_min**2
```

For a cosmological box, the physical threshold is independent of scale factor:

\[
r_{\rm num}=r_{\rm merge}\frac{L_{\rm box}/h}{2^{\ell_{\max}}}.
\]

The tree FOF implementation is transitive. Consequently, three or more sinks
can collapse into one group even when not every pair passes the direct distance
test. Such groups must be logged as `MULTIPLE`, not interpreted as an ordered
sequence of binary coalescences.

## Relative-velocity gate

When `vrel_merge=.true.`, the source computes COM-frame kinetic energy and
accepts a link if

```fortran
ekin < msink(i) * msink(j) * factG / rr
```

where `rr` is explicitly the **squared** separation. This is not the standard
two-body binding-energy expression, which scales as `1/sqrt(rr)`. The same
`1/rr` expression occurs in the sequential, serial-tree, and OpenMP tree paths.
It should therefore be treated as a legacy code proxy pending a dedicated
dimension/scaling test; it must not be described as a verified physical bound
criterion.

The Python helper `legacy_source_binding_proxy` reproduces the source
expression for regression comparisons, while `physically_bound_pair` implements
the conventional Newtonian check in physical units.

## Information loss

After grouping, the routine replaces each group with one mass-weighted COM
sink, rewrites the global sink arrays, and removes obsolete sink particles.
The current `Found N groups from M` message is not an event catalogue. Exact
member positions, velocities, IDs, and local environment must be written before
the group-accumulation/compaction loop.

## True-time interpretation

The physical coalescence time is

\[
t_{\rm true}=t_{\rm sink}+\Delta t_{r_{\rm num}\rightarrow1\,pc}
+\Delta t_{1\,pc\rightarrow0.01\,pc}^{\rm FDM}
+\Delta t_{\rm GW}.
\]

The `delay_budget` module refuses to return `t_true` while any segment is
missing, invalid, or censored. In particular, the current FDM ODE result cannot
silently stand in for the unresolved kpc-to-pc interval.
