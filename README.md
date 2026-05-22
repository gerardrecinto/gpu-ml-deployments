# GPU App Deployments

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-GPU%20Workloads-326CE5?logo=kubernetes&logoColor=white)
![CUDA 11.4](https://img.shields.io/badge/CUDA-11.4-76B900?logo=nvidia&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)

![Demo](docs/assets/demo.gif)

Containerized PyTorch training workloads deployed on Kubernetes with GPU scheduling, plus Slurm support for HPC clusters.

## Files

| File | Purpose |
|---|---|
| `train.py` | PyTorch training loop with argparse — runs on CUDA or CPU |
| `Dockerfile` | CUDA 11.4 + cuDNN 8 image with pinned torch 1.12.1 |
| `pytorch-gpu-deployment.yaml` | K8s Deployment: `nvidia.com/gpu: 1` per pod, resource requests + limits |
| `pytorch_job.sh` | Slurm batch job script |
| `gres.conf` / `slurm.conf` | Slurm GPU resource config |

## Running on Kubernetes

**1. Install the NVIDIA device plugin:**

```bash
# GPU Operator (recommended for production)
kubectl create -f https://operatorhub.io/install/nvidia-gpu-operator.yaml

# or lightweight daemonset
kubectl apply -f https://github.com/NVIDIA/k8s-device-plugin/raw/main/deployments/k8s-device-plugin-daemonset.yaml
```

**2. Build and push the image:**

```bash
docker build -t gerardrecinto/pytorch-gpu:latest .
docker push gerardrecinto/pytorch-gpu:latest
```

**3. Deploy:**

```bash
kubectl create namespace gpu-workloads
kubectl apply -f pytorch-gpu-deployment.yaml
kubectl get pods -n gpu-workloads -l app=pytorch-gpu-app
kubectl logs -f <pod-name> -n gpu-workloads
```

## Running on Slurm (HPC)

```bash
sbatch pytorch_job.sh
squeue -u $USER
```

The `gres.conf` declares GPU resources per node. `slurm.conf` sets `GresTypes`.

## CI/CD Pipeline

The `Jenkinsfile` at the repo root uses [groovylibrary](https://github.com/gerardrecinto/groovylibrary) to:

1. Build the Docker image on commit to `main`
2. Push to registry with commit SHA and `latest` tags
3. Apply the K8s manifest to the `gpu-workloads` namespace
4. Wait on rollout status before marking the build green

```bash
# Restart a running deployment manually
kubectl rollout restart deployment/pytorch-gpu-deployment -n gpu-workloads
```

## Notes

- `train.py` falls back to CPU automatically if no GPU is detected (`torch.cuda.is_available()`)
- Each pod requests 1 GPU — you need at least 2 GPU nodes for `replicas: 2`
- For multi-node distributed training, replace `pytorch-gpu-deployment.yaml` with a PyTorchJob using the `torch.distributed` launcher
