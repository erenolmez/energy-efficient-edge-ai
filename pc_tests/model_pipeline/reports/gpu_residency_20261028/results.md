# Measured safe GPU residency

NVIDIA GeForce RTX 2060; batch1, tinyrouter also resident; 1GiBfree reserve.

| Loading order | Usable classifier engines | GPU used (MiB) | GPU free after verification (MiB) |
|---|---:|---:|---:|
| smallest_first | 53 | 4915 | 1229 |
| largest_first | 12 | 5005 | 1139 |
| architecture_round_robin | 18 | 4757 | 1387 |

Demonstrated safe counts under frozen orders/reserve; not absolute maximum or speedup.

All retained engines were kept alive and re-executed with synthetic-input parity. Counts vary with model mix, context allocation, other GPU consumers and the safety reserve. No classifier accuracy or routing latency/energy saving is established by this capacity test.

The stopped multi-day baseline remains preserved and must not be resumed.
