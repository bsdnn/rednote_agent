.PHONY: dev-backend dev-frontend build install lint

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install

dev-backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build
	@echo "✅ Frontend built to backend/static/"
	uvicorn backend.main:app --host 0.0.0.0 --port 8000

lint:
	cd frontend && npm run lint
