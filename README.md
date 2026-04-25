# ktc-predictor-dev

Model development sandbox for KTC value prediction experiments.

## Structure

- `backend/` — FastAPI backend + model training code
- `frontend/` — Next.js frontend for visualizing results

## Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 5002
```

**Frontend:**
```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```
