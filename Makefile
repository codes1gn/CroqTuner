PROJECT_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
NODE_DIR     := /home/albert/local/node-v20.11.1-linux-x64
VENV         := $(PROJECT_ROOT)/backend/.venv
BACKEND_PORT := 8642
FRONTEND_PORT:= 5173
SCREEN_NAME  := croqtuner

export PATH := $(NODE_DIR)/bin:$(VENV)/bin:$(PATH)

# ── blocking / preview mode (Ctrl-C to stop) ──────────────────────────
.PHONY: serve
serve:
	@echo "Starting CroqTuner (blocking mode)…"
	@echo "  Backend  → http://0.0.0.0:$(BACKEND_PORT)"
	@echo "  Frontend → http://0.0.0.0:$(FRONTEND_PORT)"
	@trap 'kill 0' EXIT; \
	cd $(PROJECT_ROOT)/backend && $(VENV)/bin/uvicorn app.main:app \
		--host 0.0.0.0 --port $(BACKEND_PORT) & \
	cd $(PROJECT_ROOT)/frontend && $(NODE_DIR)/bin/npx vite --host 0.0.0.0 --port $(FRONTEND_PORT) & \
	wait

# ── persistent mode via GNU screen ────────────────────────────────────
.PHONY: serve-persistent
serve-persistent: _kill-screen
	@echo "Launching CroqTuner in screen session '$(SCREEN_NAME)'…"
	screen -dmS $(SCREEN_NAME) bash -c '\
		export PATH=$(NODE_DIR)/bin:$(VENV)/bin:$$PATH; \
		cd $(PROJECT_ROOT)/backend && \
		$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(BACKEND_PORT) 2>&1 | \
		while IFS= read -r line; do echo "[backend] $$line"; done & \
		BACKEND_PID=$$!; \
		sleep 2; \
		cd $(PROJECT_ROOT)/frontend && \
		$(NODE_DIR)/bin/npx vite --host 0.0.0.0 --port $(FRONTEND_PORT) 2>&1 | \
		while IFS= read -r line; do echo "[frontend] $$line"; done & \
		FRONTEND_PID=$$!; \
		trap "kill $$BACKEND_PID $$FRONTEND_PID 2>/dev/null; exit" INT TERM; \
		wait'
	@echo "Screen session '$(SCREEN_NAME)' started."
	@echo "  Backend  → http://0.0.0.0:$(BACKEND_PORT)"
	@echo "  Frontend → http://0.0.0.0:$(FRONTEND_PORT)"
	@echo "  Attach   → screen -r $(SCREEN_NAME)"

.PHONY: stop
stop: _kill-screen
	@echo "CroqTuner screen session stopped."

.PHONY: status
status:
	@screen -ls $(SCREEN_NAME) 2>/dev/null || echo "No active $(SCREEN_NAME) screen session."
	@echo ""
	@echo "Backend  (port $(BACKEND_PORT)):"
	@curl -sf http://localhost:$(BACKEND_PORT)/api/health | python3 -m json.tool 2>/dev/null || echo "  not responding"
	@echo ""
	@echo "Frontend (port $(FRONTEND_PORT)):"
	@curl -sf http://localhost:$(FRONTEND_PORT)/ >/dev/null 2>&1 && echo "  responding OK" || echo "  not responding"

.PHONY: logs
logs:
	@screen -r $(SCREEN_NAME) 2>/dev/null || echo "No active session. Start with: make serve-persistent"

# ── internal helpers ───────────────────────────────────────────────────
.PHONY: _kill-screen
_kill-screen:
	@screen -ls $(SCREEN_NAME) 2>/dev/null | grep -q '$(SCREEN_NAME)' && \
		screen -S $(SCREEN_NAME) -X quit 2>/dev/null && \
		echo "Stopped previous $(SCREEN_NAME) session." || true
	@-pkill -f "uvicorn app.main:app.*--port $(BACKEND_PORT)" 2>/dev/null || true
	@-pkill -f "vite.*--port $(FRONTEND_PORT)" 2>/dev/null || true
	@sleep 1
