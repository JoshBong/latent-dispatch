# GCP GPU search

`provision_gcp_gpu.py` uses Python 3 and the authenticated `gcloud` CLI. Google
Cloud Shell already has both. It provisions **one on-demand T4 on n1-standard-4**
with a 100 GB Ubuntu 22.04 boot disk and an ephemeral external IP. GPU drivers
and the project environment still need to be installed after provisioning.

First inspect existing resources (this does not modify anything):

```bash
python3 scripts/provision_gcp_gpu.py --project deep-learning-514621 --network deeplearning --inventory
```

Review existing VMs, attached/unattached disks, and reserved addresses before
deleting anything. The script never deletes existing resources. Inventory prints
project resource metadata; avoid posting its complete output publicly.

Start the search:

```bash
python3 scripts/provision_gcp_gpu.py --project deep-learning-514621 --network deeplearning
```

It discovers all zones advertising T4 support and attempts them sequentially,
repeating after 60 seconds until one VM is running. To limit locations or retries:

```bash
python3 scripts/provision_gcp_gpu.py --project deep-learning-514621 --network deeplearning \
  --regions us-east1,us-central1,us-west1 --max-rounds 3 --retry-seconds 120
```

The VPC must already exist. In each region the script reuses a regular IPv4 subnet
or creates `gpu-search-REGION`. New /24 ranges come from `10.200.0.0/16`, excluding
primary and secondary ranges already in the same VPC. If your VPC connects to
another network, choose `--subnet-pool` to avoid those remote ranges too. New
subnets remain after unsuccessful attempts and are reused on subsequent rounds.
No firewall rules are created or changed.

Capacity errors advance to the next zone. Regional quota errors skip the rest of
that region for the current round. Global GPU quota and unexpected errors stop
the script. If every region hits quota, it stops rather than endlessly retrying.
Provisioning requires appropriate IAM permissions, enabled Compute Engine API,
billing, global GPU quota, and regional T4/CPU quota.

The script refuses to run if its VM name already exists or has a pending insert
operation. After a creation error it checks for an existing VM before proceeding.
Run only **one copy per project/name**; this is not a distributed lock. If you
interrupt a request, it can still complete in GCP: inventory before restarting.

The successful VM starts billing immediately. An idle VM still costs money.
Stopping it releases GPU capacity; a later start can encounter another shortage.
Disks can continue billing while a VM is stopped.

For SSH, the script prints an IAP command. That needs IAP/SSH IAM permissions and
a firewall rule allowing TCP 22 from IAP's `35.235.240.0/20` range to the VM's
`gpu-worker` tag (or another applicable existing rule). For example, if no such
rule exists:

```bash
gcloud compute firewall-rules create gpu-worker-ssh-iap \
  --project deep-learning-514621 --network deeplearning \
  --direction INGRESS --action ALLOW --rules tcp:22 \
  --source-ranges 35.235.240.0/20 --target-tags gpu-worker
```

Alternatively, use an existing SSH firewall rule and the external IP. Allowing
HTTPS alone does not enable SSH. The VM has no attached service account; log in
explicitly if your workload needs authenticated Google APIs.

Local verification (no GCP access or third-party Python packages required):

```bash
python3 -m unittest discover -s scripts -p 'test_*.py'
```

## A100 40GB / Qwen3-8B masking pilot

Use the experiment code from results/t4-masking-2026-11-15 (96ba005),
with scripts/run_masking_pilot.py copied into that checkout. The runner preserves
seed 0, 100 toy + 100 MuSiQue sources, all three composition probes, FP16,
and all five masking conditions at AA/AB/BA/BB. Composition labels are recomputed
for 8B; eligible counts may differ from 4B. It does not run expanded H1/H2.
The current main checkout does not contain the required masking notebook.

Provision one A100 40GB with a GCP-enforced nine-hour stop limit:

```bash
python3 scripts/provision_gcp_gpu.py --project deep-learning-514621 \
  --network deeplearning --name latent-dispatch-a100 --gpu a100 \
  --regions us-central1,us-east1,us-west1 --max-rounds 1 --max-run-duration 9h
```

A100 is bundled in a2-highgpu-1g; no separate accelerator flag is used.
Quota and capacity are separate requirements. As checked on 2026-11-15,
NVIDIA_A100_GPUS and PREEMPTIBLE_NVIDIA_A100_GPUS are zero in every project region.
Request NVIDIA_A100_GPUS quota of 1 in a chosen region before provisioning.
Keep searches limited to regions with a reviewed price under the credit budget.
At the Iowa list price of $3.673385/hour, nine hours is about $33.06 plus disk/IP.
The runtime limit includes setup and downloads, and leaves results on the disk.
It is not a dollar spending cap. Configure guest shutdown on job exit before
running the pilot; the provisioning script alone does not stop on job completion.

After installing drivers and project dependencies and passing GPU smoke checks,
run `.venv/bin/python scripts/run_masking_pilot.py` from that results checkout.
Use a detached service with a wrapper that records the exit code and powers off
on success or failure; download results before restarting or deleting the VM.
The results bundle is results/Qwen3-8B-float16-masking-only.tar.gz.
