tags:: #tools, #devops, #containers
type:: tool
created:: [[2024-01-23]]

- Docker is a containerization platform
  - Package applications with dependencies
  - Run consistently across environments
  - Lightweight alternative to VMs
- Core components
  id:: 65a2c1f0-4444-4567-89ab-cdef0123456e
  - Images
    - Read-only templates
    - Built from Dockerfile
  - Containers
    - Running instances of images
    - Isolated processes
  - Volumes
    - Persistent data storage
  - Networks
    - Container communication
- Dockerfile
  - Instructions to build an image
  - FROM: base image
  - RUN: execute commands
  - COPY: add files
  - CMD: default command
  - EXPOSE: document ports
- Docker Compose
  - Multi-container applications
  - YAML configuration file
  - Service orchestration
    - Define services, networks, volumes
- Common use cases
  - Development environments
  - Microservices architecture
  - CI/CD pipelines
  - Application deployment
- Best practices
  - Use official base images
  - Minimize layer count
  - Don't run as root
  - Use .dockerignore
  - Multi-stage builds for smaller images
    priority:: high
