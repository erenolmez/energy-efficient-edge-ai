# Progress update for 13 October 2026

Five-page update to the supplied previous progress report. Styling follows the
tracked GitHub LaTeX report in `progress_reports/2026-10-01`, with general
introductory sections removed. Numerical cutoff: saved results on 12 October.

- `progress_report.tex`: standalone editable LaTeX, embedded tables/figure.
- `progress_report.pdf`: generated directly from the LaTeX source.
- `numerical_results.json`: detailed PC numerical supplement, including partial
  scalar/legacy tests, static precision results, repeats and source hashes.
  This 31 MB local supplement is deliberately not committed to GitHub.
- `numerical_summary.json`: published compact per-configuration numerical
  summary and source hashes, without large repeat arrays or model artifacts.
- `build_numerical_snapshot.py`: read-only experiment-data extraction; runs no GPU tests.

All 120 finished correctness configurations are tabulated inside the report.
The 1,941 partial legacy and 52 stopped global scalar routes are summarized by
family inside the page limit and retained individually in the supplement.
The paused Jetson's remote per-target accuracy/cost files are not locally
available; only verified count/hash checkpoints are reported, explicitly.
No experiments or remote devices were modified. The 1 October report is unchanged.
The five-page revision was approved for publication by the user on 13 October.

The LaTeX is standalone and can be compiled with a normal TeX installation or
Tectonic. `verify_report.py` additionally validates the 120 matrix cells against
the local full supplement. Regenerating the full supplement requires the
original local experiment reports/checkpoints, which are not bundled here.
