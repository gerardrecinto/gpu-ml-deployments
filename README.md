# GPU App Deployments

Containerized PyTorch training workloads deployed on Kubernetes with GPU scheduling, plus Slurm support for HPC clusters.

## What's in here

| File | Purpose |
|---|---|
| `train.py` | PyTorch training loop (CUDA-aware) |
| `Dockerfile` | CUDA 11.4 + cuDNN 8 runtime image |
| `pytorch-gpu-deployment.yaml` | K8s Deployment requesting `nvidia.com/gpu: 1` per pod |
| `pytorch_job.sh` | Slurm batch job script |
| `gres.conf` / `slurm.conf` | Slurm GPU resource config |

## Running on Kubernetes

**1. Install the NVIDIA device plugin** so K8s can schedule GPU resources:

```bash
# option A - NVIDIA GPU Operator (recommended for production)
kubectl create -f https://operatorhub.io/install/nvidia-gpu-operator.yaml

# option B - lightweight device plugin daemonset
kubectl apply -f https://github.com/NVIDIA/k8s-device-plugin/raw/main/deployments/k8s-device-plugin-daemonset.yaml
```

**2. Build and push the image:**

```bash
docker build -t gerardrecinto/pytorch-gpu:latest .
docker push gerardrecinto/pytorch-gpu:latest
```

**3. Deploy:**

```bash
kubectl apply -f pytorch-gpu-deployment.yaml
kubectl get pods -l app=pytorch-gpu-app
kubectl logs -f <pod-name>
```

## Running on Slurm (HPC)

```bash
sbatch pytorch_job.sh
squeue -u $USER
```

The `gres.conf` declares the GPU resources available per node. `slurm.conf` sets the GresTypes.

## CI/CD Pipeline

The `Jenkinsfile` at the repo root handles:

1. Build the Docker image on commit to `main`
2. Push to the registry with commit SHA and `latest` tags
3. Apply the K8s manifest to the `gpu-workloads` namespace
4. Wait on rollout status before marking the build green

```bash
# trigger manually if needed
kubectl rollout restart deployment/pytorch-gpu-deployment -n gpu-workloads
```

## Notes

- The training loop in `train.py` will fall back to CPU automatically if no GPU is detected (`torch.cuda.is_available()`)
- Replicas are set to 2 in the deployment but each pod requests 1 GPU, so you need at least 2 GPU nodes
- For multi-node distributed training, swap `pytorch-gpu-deployment.yaml` for a `pytorch-job.yaml` using the `torch.distributed` launcher
