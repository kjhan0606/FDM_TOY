# Native lagRamses v2 capture fixture

This minimal fixture preserves the capture ledger and checkpoint markers from
the successful two-stage serial run
`/home/kjhan/BACKUP/smbh-capture-restart-7632175-20261004`. The producing
lagRamses source commit was `7632175`. No RAMSES mesh, particle, hydro, sink,
or gravity snapshot payload is included.

The fixture exercises the native v2 batch, per-axis periodic-box metadata,
post-compaction commit, and restart-lineage contracts. It contains one
two-member `BINARY` event and one three-member `MULTIPLE` event.
