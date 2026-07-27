#!/usr/bin/env bash
# List Docker images and containers sorted by real disk size (largest first).
# Usage: sh list_docker_sizes.sh

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found"
    exit 1
fi

human_bytes() {
    awk -v b="$1" 'BEGIN {
        if (b >= 1073741824) printf "%.1fGB", b/1073741824
        else if (b >= 1048576) printf "%.1fMB", b/1048576
        else if (b >= 1024) printf "%.1fKB", b/1024
        else printf "%dB", b
    }'
}

hr() { printf '\n%s\n' "================================================================"; }

hr
echo "DOCKER DISK SUMMARY"
hr
docker system df

hr
echo "IMAGES (largest first, by bytes)"
printf '%-12s  %-45s  %s\n' "SIZE" "IMAGE" "ID"
hr

if [ "$(docker images -q | wc -l)" -eq 0 ]; then
    echo "(no images)"
else
    docker images -q | sort -u | xargs -r docker inspect --format '{{.Size}}|{{if .RepoTags}}{{index .RepoTags 0}}{{else}}<none>:<none>{{end}}|{{.Id}}' \
        | sort -t'|' -k1 -nr \
        | while IFS='|' read -r bytes name id; do
            size_human=$(human_bytes "$bytes")
            short_id=${id#sha256:}
            short_id=${short_id:0:12}
            printf '%-12s  %-45s  %s\n' "$size_human" "$name" "$short_id"
        done
fi

hr
echo "CONTAINERS (writable layer, largest first)"
printf '%-12s  %-28s  %-14s  %s\n' "SIZE" "NAME" "STATUS" "IMAGE"
hr

if [ "$(docker ps -aq | wc -l)" -eq 0 ]; then
    echo "(no containers)"
else
    docker ps -aq | xargs -r docker inspect --format '{{.SizeRw}}|{{.Name}}|{{.State.Status}}|{{.Config.Image}}' \
        | sort -t'|' -k1 -nr \
        | while IFS='|' read -r bytes name status image; do
            name="${name#/}"
            size_human=$(human_bytes "$bytes")
            printf '%-12s  %-28s  %-14s  %s\n' "$size_human" "$name" "$status" "$image"
        done
fi

hr
echo "DANGLING IMAGES (<none>:<none>) — remove with: docker image prune -f"
hr
docker images -f "dangling=true" --format '{{.Size}}\t{{.ID}}' 2>/dev/null || true

hr
echo "Tips:"
echo "  docker image prune -f          # delete dangling <none> images"
echo "  docker container prune -f      # delete stopped containers"
echo "  docker system prune -a -f      # delete all unused images/containers"
hr
