web: cd backend && LOKY_MAX_CPU_COUNT=1 gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 15
