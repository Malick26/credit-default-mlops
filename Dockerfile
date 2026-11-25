# Production image: the web app and model inference only.
# The model is NOT baked in: the app fetches the approved champion from the registry at startup.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /srv
RUN useradd --create-home --uid 1000 app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Commit the image was built from, shown by /model-info.
ARG GIT_SHA=unknown
ENV APP_GIT_SHA=${GIT_SHA}

USER app
EXPOSE 8000
# The platform gives the port in $PORT (Render, Cloud Run...).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
