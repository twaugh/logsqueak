tags:: #devops, #orchestration
type:: platform
created:: [[2024-01-24]]

- Kubernetes (K8s) orchestrates containerized applications
  - Originally developed by Google
  - Now maintained by CNCF
  - Works with [[Docker]] and other runtimes
- Architecture
  id:: 65a2c1f0-5555-4567-89ab-cdef0123456f
  - Control plane
    - API server
    - Scheduler
    - Controller manager
    - etcd (distributed key-value store)
  - Worker nodes
    - Kubelet (node agent)
    - Container runtime
    - kube-proxy
- Key concepts
  - Pod: smallest deployable unit
    - One or more containers
  - Service: stable network endpoint
  - Deployment: declarative updates
  - ConfigMap: configuration data
  - Secret: sensitive information
- Workload types
  - Deployments for stateless apps
  - StatefulSets for stateful apps
  - DaemonSets for node-level tasks
  - Jobs for batch processing
  - CronJobs for scheduled tasks
- Networking
  - Pod-to-pod communication
  - Service discovery
  - Ingress for external access
  - Network policies for security
- Storage
  - Persistent Volumes (PV)
  - Persistent Volume Claims (PVC)
  - Storage Classes
- When to use Kubernetes
  - Large-scale deployments
  - Need for auto-scaling
  - High availability requirements
  - Multi-cloud strategy
