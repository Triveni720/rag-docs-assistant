# Kubernetes basics

Kubernetes is an open-source platform for running containerised applications across a cluster of machines.

## Pods and Deployments

A Pod is the smallest deployable unit and runs one or more containers. A Deployment declares how many replicas of a Pod should run, and Kubernetes keeps that number running. If a Pod is deleted or crashes, the Deployment creates a new one automatically, which is called self-healing.

## Services

A Service gives a stable network address to a set of Pods and load-balances traffic between them, even when Pods are replaced.

## Probes

A readiness probe tells Kubernetes when a Pod is ready to receive traffic. A liveness probe tells Kubernetes when a Pod is unhealthy and should be restarted.

## Docker images

Kubernetes runs containers from images, usually built with a Dockerfile. An image packages the application code, runtime and libraries so it runs the same everywhere.
