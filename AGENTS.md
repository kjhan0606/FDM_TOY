# Project instructions

- Before writing or editing paper prose, read `/home/kjhan/WRITING.md` in full
  and apply every rule in that document.
- For the next manuscript revision, vary sentence length to maintain rhythm and
  choose verbs that state each physical process and inference unambiguously.
  This is a future prose-revision note and does not call for a manuscript rewrite
  during HR5 data regeneration.
- The working paper title is `Pulsar Timing Array in Fuzzy Dark Matter Model`.
- Paper prose must use astronomy, cosmology, gravitational-wave physics, and
  numerical-physics terminology. Do not use computer-science metaphors or
  software-development vocabulary in the scientific argument.
- Keep software instructions in README or developer documentation. Do not copy
  software terminology into manuscript text, captions, abstracts, or referee
  responses.
- Keep manuscript prose in the separate Overleaf repository. This repository
  contains the physical calculations, configurations, and validation material.
- Distinguish numerical sink encounters from physical SMBH coalescence. A true
  coalescence time requires the numerical-scale-to-1-pc interval, the FDM
  interval from 1 pc to 0.01 pc, and the gravitational-wave interval.
- Treat integrations that exceed the available cosmic time as censored physical
  results.
- Do not apply analytic FDM drag together with the force from a resolved FDM
  wake. The two forces describe the same energy and momentum exchange.
- `/home/kjhan/BACKUP/lagRamses` is a separate working tree with existing user
  changes. Do not modify that checkout unless the user explicitly requests a
  lagRamses change.

# Operational safety

- Do not run long-lived CPU post-processing through a Codex exec/wait polling
  loop. Repeated process polling previously triggered a `pgrep` process storm
  and overloaded the shared node.
- Do not use `pgrep`, `pkill`, or repeated shell process scans to monitor FDM
  work. Use a single bounded status check only when the user requests one.
- Treat 512-cubed wave-response analysis as a heavy job: it used about 37 GB of
  resident memory per process. Do not launch it, parallelize it, or resume it
  without explicit user approval.
- Before any approved heavy CPU/FFT run, cap all numerical-library thread counts
  at one, use a single process, and prefer a resumable snapshot-at-a-time
  workflow with a verified memory bound.
- If a Codex-launched task causes unexpected process creation, CPU load, or
  memory pressure, stop Codex-owned work immediately and do not retry until the
  user explicitly approves a safer execution method.
