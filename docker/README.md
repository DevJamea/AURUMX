# Docker assets (Phase 9)

Dockerfile + docker-compose.yml for the VPS deployment mode
(aurumx-api / aurumx-worker / aurumx-db / aurumx-web) land in Phase 9.

Note: the `MetaTrader5` python package is Windows-only.  In a split deployment
the MT5 terminal + broker host run on a Windows host while the API/worker/db
can run in Docker — documented in docs/DEPLOYMENT.md (Phase 9).
