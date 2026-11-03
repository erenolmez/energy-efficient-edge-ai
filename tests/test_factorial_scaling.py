import unittest
from hpc.scale_full_factorial import capacity, gpu_count


class ScalingTests(unittest.TestCase):
    def node(self, name="n1", partitions="gpu_a40", state="MIXED",
             configured=4, allocated=1, cpu=0, mem=0):
        return (f"NodeName={name} State={state} Partitions={partitions} "
                f"CfgTRES=cpu=96,gres/gpu={configured} AllocTRES=cpu={cpu},gres/gpu={allocated} "
                f"CPUEfctv=96 CPUAlloc={cpu} RealMemory=500000 AllocMem={mem}")

    def test_counts_free_not_total(self):
        count, details = capacity(self.node(), ["gpu_a40"])
        self.assertEqual(count["gpu_a40"], 3)
        self.assertEqual(len(details), 1)

    def test_busy_unhealthy_and_unauthorized_nodes_excluded(self):
        lines = [self.node(state="DOWN"), self.node(state="IDLE+DRAIN"),
                 self.node(allocated=4), self.node(partitions="restricted_gpu")]
        self.assertEqual(dict(capacity("\n".join(lines), ["gpu_a40"])[0]), {})

    def test_cpu_and_memory_limit_slots(self):
        self.assertEqual(capacity(self.node(cpu=92), ["gpu_a40"])[0]["gpu_a40"], 1)
        self.assertEqual(capacity(self.node(mem=490000), ["gpu_a40"])[0]["gpu_a40"], 0)

    def test_overlapping_partitions_count_once(self):
        count, details = capacity(self.node(partitions="gpu_h200,gpu_a40", configured=8, allocated=6),
                                  ["gpu_h200", "gpu_a40"])
        self.assertEqual(dict(count), {"gpu_h200": 2})
        self.assertEqual(len(details), 1)

    def test_zero_gpu_or_empty_allocations(self):
        self.assertEqual(gpu_count("cpu=96,gres/gpu=4"), 4)
        self.assertEqual(gpu_count(""), 0)
        self.assertEqual(gpu_count("cpu=96"), 0)


if __name__ == "__main__":
    unittest.main()
