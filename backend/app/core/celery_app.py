"""
Celery configuration for background task processing.
Uses RabbitMQ as message broker.
"""
from celery import Celery
from app.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "immobilier_rag",
    broker=settings.rabbitmq_url,
    backend=settings.redis_url,
    include=["app.tasks.document_tasks", "app.tasks.training_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Paris",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Task routing
    task_routes={
        "app.tasks.document_tasks.*": {"queue": "documents"},
        "app.tasks.training_tasks.*": {"queue": "training"},
    },
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Concurrency
    worker_concurrency=2,
)

# Optional: Configure task priorities
celery_app.conf.task_queue_max_priority = 10
celery_app.conf.task_default_priority = 5
