"""Exercise retry and stop behavior without making cloud requests."""

import argparse
import unittest
from unittest.mock import patch

import provision_gcp_gpu as gpu


class SearchTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(project="test", name="worker", network="network",
                                  regions=None, machine_type="n1-standard-4",
                                  disk_size=100, max_rounds=2, retry_seconds=1)

    def run_search(self, errors, reconcile=None, gpu_type="t4", source_snapshot=None):
        search_args = self.args()
        search_args.gpu = gpu_type
        search_args.source_snapshot = source_snapshot
        if gpu_type == "a100":
            search_args.machine_type = "a2-highgpu-1g"
            search_args.max_run_duration = "9h"
        creates = []
        checks = 0
        success = False

        def cloud(project, *args):
            nonlocal checks, success
            if args[:3] == ("compute", "instances", "list"):
                checks += 1
                if reconcile and checks > 1:
                    return reconcile()
                return ([{"name": "worker", "zone": creates[-1], "status": "RUNNING"}]
                        if success else [])
            if args[:3] == ("compute", "accelerator-types", "list"):
                return [{"name": "nvidia-tesla-a100" if gpu_type == "a100" else "nvidia-tesla-t4", "zone": zone}
                        for zone in ("us-east1-b", "us-east1-c", "us-west1-a")]
            if args[:3] == ("compute", "instances", "create"):
                if source_snapshot:
                    self.assertIn("--source-snapshot="+source_snapshot, args)
                    self.assertFalse(any(a.startswith("--image-") for a in args))
                if gpu_type == "a100":
                    self.assertIn("--machine-type=a2-highgpu-1g", args)
                    self.assertIn("--max-run-duration=9h", args)
                    self.assertIn("--instance-termination-action=STOP", args)
                    self.assertIn("--no-restart-on-failure", args)
                    self.assertFalse(any(a.startswith("--accelerator=") for a in args))
                creates.append(next(a.split("=", 1)[1] for a in args if a.startswith("--zone=")))
                error = errors.pop(0)
                if error:
                    raise gpu.CloudError(error)
                success = True
                return []
            return []

        with patch.object(gpu, "cloud", side_effect=cloud), \
             patch.object(gpu, "ensure_subnet", return_value="subnet"), \
             patch.object(gpu.time, "sleep"), patch("builtins.print"):
            try:
                gpu.provision(search_args)
            except gpu.CloudError:
                return creates, False
        return creates, True

    def test_snapshot_preserves_environment_without_fresh_image(self):
        creates, success = self.run_search([None], source_snapshot="saved-environment")
        self.assertTrue(success)
        self.assertEqual(len(creates), 1)

    def test_a100_uses_bundled_gpu_and_runtime_limit(self):
        creates, success = self.run_search(["ZONE_RESOURCE_POOL_EXHAUSTED", None], gpu_type="a100")
        self.assertTrue(success)
        self.assertEqual(creates, ["us-east1-b", "us-east1-c"])

    def test_capacity_then_success_stops(self):
        creates, success = self.run_search(["ZONE_RESOURCE_POOL_EXHAUSTED", None])
        self.assertTrue(success)
        self.assertEqual(creates, ["us-east1-b", "us-east1-c"])

    def test_regional_quota_skips_other_zone(self):
        creates, success = self.run_search(["Quota NVIDIA_T4_GPUS exceeded", None])
        self.assertTrue(success)
        self.assertEqual(creates, ["us-east1-b", "us-west1-a"])

    def test_capacity_retries_next_round(self):
        creates, success = self.run_search(["ZONE_RESOURCE_POOL_EXHAUSTED"] * 3 + [None])
        self.assertTrue(success)
        self.assertEqual(len(creates), 4)

    def test_global_quota_and_ambiguous_errors_stop(self):
        for error in ("Quota GPUS_ALL_REGIONS exceeded", "connection lost"):
            with self.subTest(error=error):
                creates, success = self.run_search([error])
                self.assertFalse(success)
                self.assertEqual(len(creates), 1)

    def test_created_vm_after_error_stops(self):
        creates, success = self.run_search(
            ["ZONE_RESOURCE_POOL_EXHAUSTED"],
            reconcile=lambda: [{"name": "worker", "zone": "us-east1-b", "status": "PROVISIONING"}],
        )
        self.assertFalse(success)
        self.assertEqual(len(creates), 1)

    def test_failed_reconciliation_stops(self):
        def failed():
            raise gpu.CloudError("Permission denied")
        creates, success = self.run_search(["ZONE_RESOURCE_POOL_EXHAUSTED"], reconcile=failed)
        self.assertFalse(success)
        self.assertEqual(len(creates), 1)

    def test_cidr_excludes_primary_and_secondary_ranges(self):
        subnets = [{"ipCidrRange": "10.200.0.0/24", "secondaryIpRanges": [
            {"ipCidrRange": "10.200.1.0/24"}]}]
        self.assertEqual(gpu.free_cidr(subnets, "10.200.0.0/16"), "10.200.2.0/24")
        with self.assertRaises(gpu.CloudError):
            gpu.free_cidr(subnets, "10.200.0.0/24")

    def test_existing_subnet_reused(self):
        args = self.args()
        with patch.object(gpu, "cloud", side_effect=[
            {"subnetworks": ["regions/us-east1/subnetworks/existing"]},
            {"name": "existing", "network": "networks/network", "region": "regions/us-east1",
             "ipCidrRange": "10.0.0.0/24"}]) as cloud:
            self.assertEqual(gpu.ensure_subnet(args, "us-east1"), "existing")
            self.assertEqual(cloud.call_count, 2)
            self.assertIn("--region=us-east1", cloud.call_args.args)

    def test_missing_subnet_created_with_unused_range(self):
        args = self.args()
        args.subnet_prefix = "gpu-search"
        args.subnet_pool = "10.200.0.0/16"
        with patch.object(gpu, "cloud", side_effect=[
            {"subnetworks": ["regions/us-west1/subnetworks/other-region"]},
            {"name": "other-region", "network": "networks/network",
             "region": "regions/us-west1", "ipCidrRange": "10.200.0.0/24"},
            {}]) as cloud, patch("builtins.print"):
            self.assertEqual(gpu.ensure_subnet(args, "us-east1"), "gpu-search-us-east1")
            self.assertIn("--range=10.200.1.0/24", cloud.call_args.args)
            self.assertIn("--region=us-east1", cloud.call_args.args)

    def test_unreadable_existing_subnet_stops_allocation(self):
        args = self.args()
        with patch.object(gpu, "cloud", side_effect=[
            {"subnetworks": ["regions/us-west1/subnetworks/existing"]},
            gpu.CloudError("regional read failed"),
        ]) as cloud:
            with self.assertRaises(gpu.CloudError):
                gpu.ensure_subnet(args, "us-east1")
            self.assertFalse(any("create" in c.args for c in cloud.call_args_list))

    def test_inventory_only_reads(self):
        with patch.object(gpu, "cloud", return_value=[]) as cloud, patch("builtins.print"):
            gpu.inventory(self.args())
            self.assertEqual(cloud.call_count, 7)
            for call in cloud.call_args_list:
                self.assertIn(call.args[-1], ("list", "describe"))


if __name__ == "__main__":
    unittest.main()
