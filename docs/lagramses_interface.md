# Future lagRamses interface

lagRamses should write a merger-event record immediately before `merge_sink`
compacts the FOF group. Snapshot differencing cannot recover the exact event
state.

Required record groups are:

- provenance: commit, binary hash, job/run ID, step, scale factor, cosmic time;
- binary: member IDs, dynamical/BH masses, positions, velocities, spins,
  separation, relative energy, and angular momentum;
- environment: gas, stellar, and FDM density/velocity information plus halo and
  fitted soliton properties.

Groups with more than two sinks must retain the complete member list and carry
a `MULTIPLE` flag. They must not be converted into an arbitrary sequence of
pair mergers.

v0.1 consumes standalone YAML cases only and does not modify lagRamses
dynamics.
