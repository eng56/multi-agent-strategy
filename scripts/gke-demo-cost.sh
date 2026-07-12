#!/usr/bin/env bash
set -euo pipefail

DEPLOYMENTS=(
  orchestrator-api
  planner-agent
  worker-agents
  verifier-agent
  skeptic-agent
  aggregator-agent
  judge-agent
  tool-runner
)

usage() {
  cat <<'USAGE'
Usage: scripts/gke-demo-cost.sh sleep|wake|status

sleep   Scale the demo deployments to 0 replicas to stop Autopilot pods while idle.
wake    Scale the demo deployments back to 1 replica and wait for rollouts.
status  Show deployment, pod, and orchestrator LoadBalancer status.

This does not delete the GKE cluster or the orchestrator LoadBalancer service.
For the lowest idle cost, delete the cluster when the demo is not needed.
USAGE
}

resources=()
for deployment in "${DEPLOYMENTS[@]}"; do
  resources+=("deployment/${deployment}")
done

command="${1:-status}"
case "${command}" in
  sleep)
    kubectl scale "${resources[@]}" --replicas=0
    echo "Scaled demo deployments to zero replicas. The GKE cluster and LoadBalancer may still accrue fixed charges."
    ;;
  wake)
    kubectl scale "${resources[@]}" --replicas=1
    for deployment in "${DEPLOYMENTS[@]}"; do
      kubectl rollout status "deployment/${deployment}" --timeout=10m
    done
    ;;
  status)
    kubectl get "${resources[@]}"
    kubectl get pods
    kubectl get svc orchestrator-api || true
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
