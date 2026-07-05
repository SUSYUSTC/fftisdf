This directory is the archived diamond-specific workflow.

It contains old scripts, logs, exploratory tests, and data layouts that were
written assuming the system is diamond.

The active workflow now lives at the top level and is system-generic:

- one system per `data_{system}` directory
- primitive structure in `{system}_prim.xyz` and `{system}_prim.lattice`
- runtime defaults such as `pseudo_potential` and `ke_cutoff` stored as small
  text files in the same data directory

Use this directory as reference material or as a source to copy from when
building a new system-specific workflow.
