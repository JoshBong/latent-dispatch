#!/usr/bin/env python3
"""Find a single on-demand T4 or A100 40GB VM using the gcloud CLI (stdlib only)."""

import argparse
import ipaddress
import json
import re
import shutil
import subprocess
import sys
import time


class CloudError(RuntimeError):
    pass


def cloud(project, *args):
    result = subprocess.run(
        ["gcloud", *args, f"--project={project}", "--quiet", "--format=json"],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise CloudError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout) if result.stdout.strip() else None


def basename(value):
    return value.rsplit("/", 1)[-1]


def matching_vms(project, name):
    # List failure must halt the search; an unreadable project is not an empty one.
    return [v for v in cloud(project, "compute", "instances", "list") if v["name"] == name]


def report_vm(project, vm):
    zone = basename(vm["zone"])
    print(f"\nVM exists: {vm['name']} in {zone}, status={vm['status']}")
    print(f"gcloud compute ssh {vm['name']} --zone={zone} --project={project} --tunnel-through-iap")
    print("SSH needs an appropriate firewall rule and IAM permissions; see scripts/README.md.")


def free_cidr(subnets, pool):
    used = []
    for subnet in subnets:
        used.append(ipaddress.ip_network(subnet["ipCidrRange"]))
        used.extend(ipaddress.ip_network(r["ipCidrRange"])
                    for r in subnet.get("secondaryIpRanges", []))
    for candidate in ipaddress.ip_network(pool).subnets(new_prefix=24):
        if not any(candidate.overlaps(existing) for existing in used):
            return str(candidate)
    raise CloudError(f"No unused /24 left in {pool}; choose another --subnet-pool.")


def ensure_subnet(args, region):
    # Use the VPC's authoritative links and explicit regional reads. An
    # aggregated subnet list can be incomplete; missing ranges must not be
    # interpreted as free address space.
    network = cloud(args.project, "compute", "networks", "describe", args.network)
    links = network.get("subnetworks", [])
    def read(link):
        return cloud(args.project, "compute", "networks", "subnets", "describe",
                     basename(link), f"--region={link.split('/regions/')[1].split('/')[0]}")
    regional = [link for link in links if f"/regions/{region}/" in link]
    loaded = {link: read(link) for link in regional}
    candidates = [s for s in loaded.values()
                  if s.get("purpose", "PRIVATE") == "PRIVATE" and s.get("ipCidrRange")]
    if candidates:
        return sorted(candidates, key=lambda s: s["name"])[0]["name"]
    for link in links:
        if link not in loaded:
            loaded[link] = read(link)
    subnets = list(loaded.values())
    name = f"{args.subnet_prefix}-{region}"
    cidr = free_cidr(subnets, args.subnet_pool)
    print(f"Creating subnet {name}: {cidr}", flush=True)
    cloud(args.project, "compute", "networks", "subnets", "create", name,
          f"--network={args.network}", f"--region={region}", f"--range={cidr}")
    return name


def failure_kind(message):
    text = message.lower()
    if "gpus_all_regions" in text or "gpus-all-regions" in text:
        return "fatal"
    if "quota" in text and ("exceeded" in text or "limit" in text):
        return "quota"
    if any(marker in text for marker in (
        "zone_resource_pool_exhausted", "resource_pool_exhausted",
        "does not have enough resources available", "resource_availability",
    )):
        return "capacity"
    return "fatal"


def inventory(args):
    for label, command in (
        ("VMs", ("compute", "instances", "list")),
        ("Disks (check users before deleting)", ("compute", "disks", "list")),
        ("Reserved addresses", ("compute", "addresses", "list")),
        ("VPC networks", ("compute", "networks", "list")),
        ("Subnets", ("compute", "networks", "subnets", "list")),
        ("Firewall rules", ("compute", "firewall-rules", "list")),
        ("Global quotas", ("compute", "project-info", "describe")),
    ):
        print(f"\n{label}:\n{json.dumps(cloud(args.project, *command), indent=2)}")


def provision(args):
    existing = matching_vms(args.project, args.name)
    if existing:
        for vm in existing:
            report_vm(args.project, vm)
        raise CloudError("Instance name already exists. Inspect it before restarting this search.")
    # Also stop on an in-flight insertion from a previously interrupted run.
    for operation in cloud(args.project, "compute", "operations", "list"):
        if (operation.get("operationType") == "insert"
                and operation.get("status") != "DONE"
                and operation.get("targetLink", "").endswith(f"/instances/{args.name}")):
            raise CloudError("An earlier VM creation is still in progress; wait and inspect it.")

    cloud(args.project, "compute", "networks", "describe", args.network)
    gpu = getattr(args, "gpu", "t4")
    accelerator = {"t4": "nvidia-tesla-t4", "a100": "nvidia-tesla-a100"}[gpu]
    types = cloud(args.project, "compute", "accelerator-types", "list",
                  f"--filter=name={accelerator}")
    zones = sorted({basename(t["zone"]) for t in types
                    if t["name"] == accelerator})
    if args.regions:
        preferred = args.regions.split(",")
        zones = sorted((z for z in zones if z.rsplit("-", 1)[0] in preferred),
                       key=lambda z: (preferred.index(z.rsplit("-", 1)[0]), z))
    if not zones:
        raise CloudError("No matching GPU zones found for the selected regions.")
    print(f"Trying {len(zones)} zones; one {gpu}, {args.machine_type}, {args.disk_size} GB disk.")
    print("A successful VM starts billing immediately. Ctrl+C stops further attempts.")
    round_number = 0
    while True:
        round_number += 1
        quota_regions = set()
        capacity_failures = 0
        for zone in zones:
            region = zone.rsplit("-", 1)[0]
            if region in quota_regions:
                continue
            subnet = ensure_subnet(args, region)
            print(f"Round {round_number}: provisioning {args.name} in {zone} ...", flush=True)
            extra = []
            if getattr(args, "max_run_duration", None):
                extra += [f"--max-run-duration={args.max_run_duration}",
                          "--instance-termination-action=STOP", "--no-restart-on-failure"]
            image = ([f"--source-snapshot={args.source_snapshot}"]
                     if getattr(args, "source_snapshot", None) else
                     ["--image-project=ubuntu-os-cloud", "--image-family=ubuntu-2204-lts"])
            try:
                cloud(args.project, "compute", "instances", "create", args.name,
                      f"--zone={zone}", f"--machine-type={args.machine_type}",
                      *([f"--accelerator=type={accelerator},count=1"] if gpu == "t4" else []),
                      f"--subnet={subnet}", "--maintenance-policy=TERMINATE",
                      "--provisioning-model=STANDARD", "--reservation-affinity=none",
                      *image,
                      f"--boot-disk-size={args.disk_size}GB", "--boot-disk-type=pd-balanced",
                      "--no-service-account", "--no-scopes", "--tags=gpu-worker",
                      "--labels=managed-by=gpu-search", *extra)
            except CloudError as error:
                print(str(error), file=sys.stderr, flush=True)
                # Even an unsuccessful CLI call may have created a VM. Never
                # proceed if reconciliation fails, or if any instance exists.
                found = matching_vms(args.project, args.name)
                if found:
                    for vm in found:
                        report_vm(args.project, vm)
                    raise CloudError("VM found after an error; stopping for inspection.")
                kind = failure_kind(str(error))
                if kind == "quota":
                    quota_regions.add(region)
                    continue
                if kind != "capacity":
                    raise CloudError("Stopping on an unexpected error; no further VM attempts.") from error
                capacity_failures += 1
                continue
            found = matching_vms(args.project, args.name)
            if len(found) != 1 or found[0]["status"] != "RUNNING":
                raise CloudError("Create returned successfully; inspect VM state before proceeding.")
            report_vm(args.project, found[0])
            return
        if not capacity_failures:
            raise CloudError("All attempted regions were quota-blocked. Fix quotas before retrying.")
        if args.max_rounds and round_number >= args.max_rounds:
            raise CloudError("Reached --max-rounds without provisioning a VM.")
        print(f"No capacity found. Retrying in {args.retry_seconds}s ...", flush=True)
        time.sleep(args.retry_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--network", default="gpu-network", help="Existing VPC name")
    parser.add_argument("--name", default="latent-dispatch-gpu")
    parser.add_argument("--regions", help="Comma-separated regions, in preference order; default: all matching GPU regions")
    parser.add_argument("--gpu", choices=["t4", "a100"], default="t4")
    parser.add_argument("--machine-type", help="Defaults to n1-standard-4 for T4 or a2-highgpu-1g for A100")
    parser.add_argument("--max-run-duration", help="GCP-enforced runtime limit, e.g. 9h; stops and preserves disks")
    parser.add_argument("--source-snapshot", help="Boot from a saved environment instead of a fresh Ubuntu image")
    parser.add_argument("--disk-size", type=int, default=100, help="Boot disk GB")
    parser.add_argument("--subnet-prefix", default="gpu-search")
    parser.add_argument("--subnet-pool", default="10.200.0.0/16", help="Allocate non-overlapping /24s here")
    parser.add_argument("--retry-seconds", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 retries indefinitely")
    parser.add_argument("--inventory", action="store_true", help="Read-only resource inventory; no provisioning")
    args = parser.parse_args()
    if args.machine_type is None:
        args.machine_type = "a2-highgpu-1g" if args.gpu == "a100" else "n1-standard-4"
    if args.gpu == "a100" and args.machine_type != "a2-highgpu-1g":
        parser.error("A100 search supports exactly one 40GB GPU: a2-highgpu-1g")
    if args.max_run_duration and not re.fullmatch(r"(?:[0-9]+[dhms])+", args.max_run_duration):
        parser.error("max-run-duration must be a duration such as 9h or 30m")
    if args.retry_seconds < 1 or args.max_rounds < 0 or args.disk_size < 10:
        parser.error("retry-seconds must be positive, max-rounds nonnegative, disk-size >= 10")
    try:
        pool = ipaddress.ip_network(args.subnet_pool)
        if pool.version != 4 or pool.prefixlen > 24 or not pool.subnet_of(ipaddress.ip_network("10.0.0.0/8")):
            raise ValueError()
    except ValueError:
        parser.error("subnet-pool must be a /24 or larger network within 10.0.0.0/8")
    for name in (args.network, args.name, args.subnet_prefix):
        if not re.fullmatch(r"[a-z](?:[-a-z0-9]*[a-z0-9])?", name) or len(name) > 63:
            parser.error(f"Invalid GCP resource name: {name}")
    if len(args.subnet_prefix) > 35:
        parser.error("subnet-prefix must be at most 35 characters")
    if not shutil.which("gcloud"):
        parser.error("gcloud not found. Run in Google Cloud Shell or install the Google Cloud CLI.")
    try:
        inventory(args) if args.inventory else provision(args)
    except CloudError as error:
        print(f"\nStopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. A submitted create may still finish. Inventory before rerunning.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
