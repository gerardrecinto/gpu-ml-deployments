# GPU App Deployments

Containerized PyTorch training workloads deployed on Kubernetes with GPU scheduling, plus Slurm support for HPC clusters.

## Files

| File | Purpose |
|---|---|
| `train.py` | PyTorch training loop with argparse — runs on CUDA or CPU |
| `Dockerfile` | CUDA 11.4 + cuDNN 8 image with pinned torch 1.12.1 |
| `pytorch-gpu-deployment.yaml` | K8s Deployment: `nvidia.com/gpu: 1` per pod, resource requests + limits |
| `pytorch_job.sh` | Slurm batch job script |
| `gres.conf` / `slurm.conf` | Slurm GPU resource config |

## Demo output

```
$ python3 train.py --epochs 10 --lr 0.01 --batch-size 64
Training on: cuda
Epochs: 10 | LR: 0.01 | Batch: 64
----------------------------------------
Epoch [ 1/10]  loss: 1.163555
Epoch [ 2/10]  loss: 1.149995
Epoch [ 3/10]  loss: 1.130859
Epoch [ 4/10]  loss: 1.106126
Epoch [ 5/10]  loss: 1.089103
Epoch [ 6/10]  loss: 1.077913
Epoch [ 7/10]  loss: 1.068456
Epoch [ 8/10]  loss: 1.050132
Epoch [ 9/10]  loss: 1.039331
Epoch [10/10]  loss: 1.015603
----------------------------------------
Training complete.
```

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
