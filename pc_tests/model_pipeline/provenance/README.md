# Structure-only source lineage

The scientific runs used the source at commit
`01b7651dd9d30d8f971ad465c1083d0d580e93fb`. The restructuring does not claim
that the organized files have the same byte hashes. It records three separate
identities for each moved file in `structure_refactor.json`:

* `original_sha256`: the historical Git blob, still available at `base_commit`;
* `canonical_sha256`: the organized implementation;
* `wrapper_sha256`: the compatibility entry point at the old location.

`canonical` is relative to the pipeline directory. The one test destination is
`../../tests/test_p80_batches.py`; no data or artifact paths are relocated.
LF source checkout rules keep these new hashes stable on Windows.

The two updated saved-result audits explicitly verify the historical source
using `git show`, and separately verify the canonical and wrapper hashes. All
data, engine, protected-report and scientific protocol checks remain exact.
They need the historical Git commit locally; a source-only ZIP or shallow clone
without it cannot establish that provenance and fails rather than skipping it.

`tests/test_pipeline_structure.py` additionally compares the historical and
organized Python ASTs after removing import/bootstrap scaffolding and
normalizing the documented filesystem anchors. Every non-audit Python move
retains the same scientific statements. The two audits are explicit exceptions:
their source checks now distinguish the historical files from the refactor.
The PowerShell implementations are moved verbatim; root wrappers only delegate.

The existing October 20 publication code manifest and all frozen experiment
protocols remain historical evidence. Do not regenerate their hashes to make a
new runner resume old checkpoints. No experiments or launchers were run for
this cleanup. Runner consolidation is a separate future change.
