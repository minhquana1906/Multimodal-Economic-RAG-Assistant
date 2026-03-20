---
name: stack-status
description: Check health of all RAG pipeline services (embedding, reranker, guard, qdrant, orchestrator)
disable-model-invocation: true
---

Run the following health checks and report status for each service:

```bash
for service in "embedding:8001" "reranker:8002" "guard:8003" "orchestrator:8000"; do
  name="${service%%:*}"; port="${service##*:}"
  status=$(curl -sf "http://localhost:$port/health" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "DOWN")
  echo "$name ($port): $status"
done
curl -sf http://localhost:6333/healthz -o /dev/null -w "qdrant (6333): %{http_code}\n" 2>/dev/null || echo "qdrant (6333): DOWN"
