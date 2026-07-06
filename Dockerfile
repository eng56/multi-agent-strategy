FROM python:3.12-slim
ARG GIT_SHA=
ARG IMAGE_TAG=
ARG BUILD_TIME=
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV GIT_SHA=${GIT_SHA} IMAGE_TAG=${IMAGE_TAG} BUILD_TIME=${BUILD_TIME}
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY entrypoint.sh /entrypoint.sh
USER appuser
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
