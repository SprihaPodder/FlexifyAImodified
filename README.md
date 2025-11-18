# FlexifyAImodified — repo instructions

This repo contains frontend and backend services. Use the Dockerfiles for deployment (Render or other providers). Follow these quick steps locally before pushing to GitHub.

Local setup (macOS):
1. Create Python venv and install backend deps (use Python 3.11):
   - python3.11 -m venv venv
   - source venv/bin/activate
   - pip install --upgrade pip
   - pip install -r backend/requirements.txt

2. Frontend:
   - cd frontend
   - npm ci
   - npm run dev (or npm run build to create dist)

3. Docker (local test):
   - docker-compose build
   - docker-compose up

Deploy to Render:
- render.yaml is provided at repo root. Push to GitHub and connect the repo in Render; Render will create two Docker services from the Dockerfiles.

Git push steps (one-time):
- See instructions below or run the commands in your terminal.
