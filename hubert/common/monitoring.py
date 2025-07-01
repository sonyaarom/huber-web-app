import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.stdlib import StdlibIntegration
import logging
from typing import Optional
from hubert.config import settings

logger = logging.getLogger(__name__)

def init_sentry() -> bool:
    """
    Initialize Sentry SDK with comprehensive monitoring for HuBer application.
    
    Returns:
        bool: True if Sentry was successfully initialized, False otherwise
    """
    if not settings.sentry_dsn:
        logger.info("Sentry DSN not configured, skipping Sentry initialization")
        return False
    
    try:
        # Configure logging integration
        logging_integration = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )
        
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            release=settings.sentry_release,
            sample_rate=settings.sentry_sample_rate,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            integrations=[
                FlaskIntegration(
                    transaction_style='endpoint'
                ),
                SqlalchemyIntegration(),
                ThreadingIntegration(
                    propagate_hub=True
                ),
                logging_integration,
                StdlibIntegration()
            ],
            # Custom tag configuration
            before_send=before_send_filter,
            before_send_transaction=before_send_transaction_filter,
            # Performance monitoring
            profiles_sample_rate=0.1,
            # Custom configuration
            max_breadcrumbs=100,
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send personally identifiable information
        )
        
        # Set custom tags for better filtering
        sentry_sdk.set_tag("component", "huber-rag")
        sentry_sdk.set_tag("service", "main")
        
        logger.info(f"Sentry initialized successfully for environment: {settings.sentry_environment}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False

def before_send_filter(event, hint):
    """
    Filter events before sending to Sentry.
    Used to exclude certain errors or add custom context.
    """
    # Don't send database connection pool exhaustion during startup
    if 'exception' in event:
        exc_info = event['exception']['values'][0]
        if 'pool pre-ping' in str(exc_info.get('value', '')):
            return None
    
    # Add custom context for RAG operations
    if event.get('transaction'):
        with sentry_sdk.configure_scope() as scope:
            scope.set_context("rag_context", {
                "embedding_model": settings.embedding_model,
                "use_reranker": settings.use_reranker,
                "use_hybrid_search": settings.use_hybrid_search,
                "top_k": settings.top_k
            })
    
    return event

def before_send_transaction_filter(event, hint):
    """
    Filter performance transactions before sending to Sentry.
    """
    # Skip health check endpoints from performance monitoring
    if event.get('transaction') in ['/health', '/ping', '/metrics']:
        return None
    
    return event

def capture_rag_operation(operation_name: str, query: str, duration: Optional[float] = None):
    """
    Capture RAG-specific operations for monitoring.
    
    Args:
        operation_name: Name of the operation (e.g., 'retrieval', 'generation')
        query: The user query being processed
        duration: Operation duration in seconds
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("operation", operation_name)
        scope.set_context("query_context", {
            "query_length": len(query),
            "query_hash": hash(query) % 10000,  # Anonymous query identifier
            "duration": duration
        })
        
        if duration:
            sentry_sdk.set_measurement(f"{operation_name}_duration", duration, "second")

def capture_retrieval_metrics(retrieved_count: int, top_k: int, scores: list):
    """
    Capture retrieval-specific metrics.
    
    Args:
        retrieved_count: Number of documents retrieved
        top_k: Configured top_k parameter
        scores: List of similarity scores
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_context("retrieval_metrics", {
            "retrieved_count": retrieved_count,
            "top_k": top_k,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0
        })
        
        # Custom measurements
        sentry_sdk.set_measurement("retrieved_documents", retrieved_count)
        sentry_sdk.set_measurement("retrieval_efficiency", retrieved_count / top_k if top_k > 0 else 0)

def capture_generation_metrics(prompt_length: int, response_length: int, duration: float):
    """
    Capture generation-specific metrics.
    
    Args:
        prompt_length: Length of the input prompt
        response_length: Length of the generated response
        duration: Generation duration in seconds
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_context("generation_metrics", {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "duration": duration,
            "tokens_per_second": response_length / duration if duration > 0 else 0
        })
        
        # Custom measurements
        sentry_sdk.set_measurement("prompt_length", prompt_length, "character")
        sentry_sdk.set_measurement("response_length", response_length, "character")
        sentry_sdk.set_measurement("generation_duration", duration, "second")

def capture_user_feedback(feedback_type: str, query: str, rating: Optional[int] = None):
    """
    Capture user feedback for quality monitoring.
    
    Args:
        feedback_type: Type of feedback ('thumbs_up', 'thumbs_down', 'rating')
        query: The query that was rated
        rating: Numerical rating if applicable
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("feedback_type", feedback_type)
        scope.set_context("feedback_context", {
            "query_hash": hash(query) % 10000,
            "rating": rating,
            "feedback_type": feedback_type
        })
        
        # Track feedback as a custom event
        sentry_sdk.add_breadcrumb(
            category='user_feedback',
            message=f'User provided {feedback_type} feedback',
            level='info',
            data={
                'feedback_type': feedback_type,
                'rating': rating
            }
        )

def capture_crawler_metrics(pages_processed: int, errors: int, duration: float):
    """
    Capture web crawler metrics.
    
    Args:
        pages_processed: Number of pages successfully processed
        errors: Number of errors encountered
        duration: Total crawling duration in seconds
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_context("crawler_metrics", {
            "pages_processed": pages_processed,
            "errors": errors,
            "duration": duration,
            "success_rate": (pages_processed / (pages_processed + errors)) if (pages_processed + errors) > 0 else 0
        })
        
        # Custom measurements
        sentry_sdk.set_measurement("pages_processed", pages_processed)
        sentry_sdk.set_measurement("crawler_errors", errors)
        sentry_sdk.set_measurement("crawler_duration", duration, "second") 