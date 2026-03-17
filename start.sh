#!/bin/bash
celery -A app.core.celery_app worker --loglevel=info &
CELERY_PID=$!
echo "Started Celery worker with PID: $CELERY_PID"
uvicorn server:app --host 0.0.0.0 --port 8000
kill $CELERY_PID