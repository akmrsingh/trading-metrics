"""
Backfill Database Module

Database helpers for backfill job queue, settings, and model-symbol configuration.
Requires a SQLAlchemy engine to be configured via set_engine().
"""

from datetime import date, timedelta
from typing import Optional, List, Dict
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Module-level engine reference
_engine: Optional[Engine] = None


def set_engine(engine: Engine):
    """Set the SQLAlchemy engine for database operations."""
    global _engine
    _engine = engine


def get_engine() -> Optional[Engine]:
    """Get the configured engine."""
    return _engine


def is_configured() -> bool:
    """Check if the database engine is configured."""
    return _engine is not None


# ============================================================
# Schema Creation
# ============================================================

BACKFILL_SCHEMA_SQL = """
-- Backfill Jobs: Queue for backfill tasks
CREATE TABLE IF NOT EXISTS backfill_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(20) NOT NULL,        -- 'prices' or 'predictions'
    symbol VARCHAR(10) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending/running/paused/completed/failed
    priority INT DEFAULT 100,             -- lower = higher priority
    progress INT DEFAULT 0,               -- 0-100
    items_total INT DEFAULT 0,            -- total days to process
    items_completed INT DEFAULT 0,        -- days processed
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    UNIQUE(job_type, symbol, start_date, end_date)
);

CREATE INDEX IF NOT EXISTS idx_backfill_jobs_status ON backfill_jobs(status);
CREATE INDEX IF NOT EXISTS idx_backfill_jobs_priority ON backfill_jobs(priority, created_at);

-- Backfill Settings: Worker configuration
CREATE TABLE IF NOT EXISTS backfill_settings (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Model-Symbol Configuration: Which model x symbol combinations to backfill
CREATE TABLE IF NOT EXISTS model_symbol_config (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(model_name, symbol)
);

CREATE INDEX IF NOT EXISTS idx_model_symbol_config_model ON model_symbol_config(model_name);
CREATE INDEX IF NOT EXISTS idx_model_symbol_config_symbol ON model_symbol_config(symbol);
"""

DEFAULT_SETTINGS = {
    'worker_enabled': 'false',
    'quiet_hours_start': '09:30',
    'quiet_hours_end': '16:00',
    'rate_limit_seconds': '2',
    'scan_days': '365',
}


def init_backfill_tables():
    """Create backfill tables if they don't exist. Uses single atomic transaction."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        # All operations in single transaction
        conn.execute(text(BACKFILL_SCHEMA_SQL))

        # Insert default settings if not present
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(text("""
                INSERT INTO backfill_settings (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key) DO NOTHING
            """), {'key': key, 'value': value})

        # Single commit at end - atomic
        conn.commit()

    return True


# ============================================================
# Backfill Settings
# ============================================================

def get_setting(key: str, default: str = None) -> Optional[str]:
    """Get a backfill setting value."""
    if not is_configured():
        return default

    with _engine.connect() as conn:
        result = conn.execute(text(
            "SELECT value FROM backfill_settings WHERE key = :key"
        ), {'key': key})
        row = result.fetchone()
        return row[0] if row else default


def get_all_settings() -> Dict[str, str]:
    """Get all backfill settings."""
    if not is_configured():
        return DEFAULT_SETTINGS.copy()

    with _engine.connect() as conn:
        result = conn.execute(text("SELECT key, value FROM backfill_settings"))
        return {row[0]: row[1] for row in result.fetchall()}


def update_setting(key: str, value: str) -> bool:
    """Update a backfill setting."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO backfill_settings (key, value, updated_at)
            VALUES (:key, :value, NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
        """), {'key': key, 'value': value})
        conn.commit()
    return True


def is_worker_enabled() -> bool:
    """Check if the backfill worker is enabled."""
    return get_setting('worker_enabled', 'false').lower() == 'true'


def set_worker_enabled(enabled: bool) -> bool:
    """Enable or disable the backfill worker."""
    return update_setting('worker_enabled', 'true' if enabled else 'false')


# ============================================================
# Model-Symbol Configuration
# ============================================================

def get_model_symbol_config() -> List[Dict]:
    """Get all model-symbol configurations."""
    if not is_configured():
        return []

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, model_name, symbol, enabled, created_at
            FROM model_symbol_config
            ORDER BY model_name, symbol
        """))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]


def get_enabled_model_symbols() -> List[Dict]:
    """Get only enabled model-symbol combinations."""
    if not is_configured():
        return []

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT model_name, symbol
            FROM model_symbol_config
            WHERE enabled = true
            ORDER BY model_name, symbol
        """))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]


def set_model_symbol_enabled(model_name: str, symbol: str, enabled: bool) -> bool:
    """Enable or disable a model-symbol combination."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO model_symbol_config (model_name, symbol, enabled)
            VALUES (:model_name, :symbol, :enabled)
            ON CONFLICT (model_name, symbol) DO UPDATE SET
                enabled = EXCLUDED.enabled
        """), {'model_name': model_name, 'symbol': symbol, 'enabled': enabled})
        conn.commit()
    return True


def update_model_symbol_config_batch(configs: List[Dict]) -> int:
    """
    Batch update model-symbol configurations.

    Args:
        configs: List of {'model_name': str, 'symbol': str, 'enabled': bool}

    Returns:
        Number of rows updated
    """
    if not is_configured():
        return 0

    count = 0
    with _engine.connect() as conn:
        for cfg in configs:
            conn.execute(text("""
                INSERT INTO model_symbol_config (model_name, symbol, enabled)
                VALUES (:model_name, :symbol, :enabled)
                ON CONFLICT (model_name, symbol) DO UPDATE SET
                    enabled = EXCLUDED.enabled
            """), cfg)
            count += 1
        conn.commit()
    return count


def ensure_model_symbol_configs(models: List[str], symbols: List[str]) -> int:
    """
    Ensure all combinations of models x symbols exist in config table.
    New combinations are created as enabled by default.

    Returns:
        Number of new configs created
    """
    if not is_configured():
        return 0

    count = 0
    with _engine.connect() as conn:
        for model_name in models:
            for symbol in symbols:
                result = conn.execute(text("""
                    INSERT INTO model_symbol_config (model_name, symbol, enabled)
                    VALUES (:model_name, :symbol, true)
                    ON CONFLICT (model_name, symbol) DO NOTHING
                    RETURNING id
                """), {'model_name': model_name, 'symbol': symbol})
                if result.fetchone():
                    count += 1
        conn.commit()
    return count


# ============================================================
# Backfill Jobs
# ============================================================

def create_job(
    job_type: str,
    symbol: str,
    start_date: date,
    end_date: date,
    priority: int = 100
) -> Optional[int]:
    """
    Create a new backfill job.

    Args:
        job_type: 'prices' or 'predictions'
        symbol: Trading symbol
        start_date: Start date for backfill
        end_date: End date for backfill
        priority: Lower = higher priority (default 100)

    Returns:
        Job ID if created, None if failed or duplicate

    Raises:
        ValueError: If validation fails (invalid job_type, dates, etc.)
    """
    if not is_configured():
        return None

    # Validation
    if job_type not in ('prices', 'predictions'):
        raise ValueError(f"Invalid job_type: {job_type}. Must be 'prices' or 'predictions'")

    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date})")

    if priority < 0:
        raise ValueError(f"Priority must be non-negative, got {priority}")

    # Calculate total trading days (weekdays)
    total_days = 0
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            total_days += 1
        current += timedelta(days=1)

    with _engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO backfill_jobs
            (job_type, symbol, start_date, end_date, priority, items_total)
            VALUES (:job_type, :symbol, :start_date, :end_date, :priority, :items_total)
            ON CONFLICT (job_type, symbol, start_date, end_date) DO NOTHING
            RETURNING id
        """), {
            'job_type': job_type,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'priority': priority,
            'items_total': total_days,
        })
        conn.commit()
        row = result.fetchone()
        return row[0] if row else None


def create_jobs_bulk(jobs: List[Dict]) -> List[int]:
    """
    Create multiple backfill jobs at once.

    Args:
        jobs: List of job dicts with job_type, symbol, start_date, end_date, priority

    Returns:
        List of created job IDs
    """
    created_ids = []
    for job in jobs:
        job_id = create_job(
            job_type=job['job_type'],
            symbol=job['symbol'],
            start_date=job['start_date'],
            end_date=job['end_date'],
            priority=job.get('priority', 100)
        )
        if job_id:
            created_ids.append(job_id)
    return created_ids


def get_job(job_id: int) -> Optional[Dict]:
    """Get a specific job by ID."""
    if not is_configured():
        return None

    with _engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM backfill_jobs WHERE id = :id"
        ), {'id': job_id})
        row = result.fetchone()
        return dict(zip(result.keys(), row)) if row else None


def get_job_queue(include_completed: bool = False) -> List[Dict]:
    """
    Get all jobs in the queue.

    Args:
        include_completed: If True, include completed/failed jobs

    Returns:
        List of job dicts, ordered by priority then created_at
    """
    if not is_configured():
        return []

    with _engine.connect() as conn:
        if include_completed:
            query = """
                SELECT * FROM backfill_jobs
                ORDER BY
                    CASE status
                        WHEN 'running' THEN 0
                        WHEN 'pending' THEN 1
                        WHEN 'paused' THEN 2
                        WHEN 'failed' THEN 3
                        WHEN 'completed' THEN 4
                    END,
                    priority ASC,
                    created_at ASC
            """
        else:
            query = """
                SELECT * FROM backfill_jobs
                WHERE status IN ('pending', 'running', 'paused')
                ORDER BY priority ASC, created_at ASC
            """
        result = conn.execute(text(query))
        return [dict(zip(result.keys(), row)) for row in result.fetchall()]


def get_next_job() -> Optional[Dict]:
    """Get the next pending job to process (highest priority)."""
    if not is_configured():
        return None

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM backfill_jobs
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
        """))
        row = result.fetchone()
        return dict(zip(result.keys(), row)) if row else None


def update_job_status(job_id: int, status: str, error_message: str = None) -> bool:
    """Update job status."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        if status == 'running':
            conn.execute(text("""
                UPDATE backfill_jobs
                SET status = :status, started_at = NOW(), error_message = NULL
                WHERE id = :id
            """), {'id': job_id, 'status': status})
        elif status in ('completed', 'failed'):
            conn.execute(text("""
                UPDATE backfill_jobs
                SET status = :status, completed_at = NOW(), error_message = :error
                WHERE id = :id
            """), {'id': job_id, 'status': status, 'error': error_message})
        else:
            conn.execute(text("""
                UPDATE backfill_jobs
                SET status = :status, error_message = :error
                WHERE id = :id
            """), {'id': job_id, 'status': status, 'error': error_message})
        conn.commit()
    return True


def update_job_progress(job_id: int, items_completed: int, progress: int = None) -> bool:
    """Update job progress."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        if progress is not None:
            conn.execute(text("""
                UPDATE backfill_jobs
                SET items_completed = :items_completed, progress = :progress
                WHERE id = :id
            """), {'id': job_id, 'items_completed': items_completed, 'progress': progress})
        else:
            # Auto-calculate progress
            conn.execute(text("""
                UPDATE backfill_jobs
                SET items_completed = :items_completed,
                    progress = CASE
                        WHEN items_total > 0 THEN (items_completed * 100 / items_total)
                        ELSE 0
                    END
                WHERE id = :id
            """), {'id': job_id, 'items_completed': items_completed})
        conn.commit()
    return True


def delete_job(job_id: int) -> bool:
    """Delete a job from the queue."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        conn.execute(text("DELETE FROM backfill_jobs WHERE id = :id"), {'id': job_id})
        conn.commit()
    return True


def clear_pending_jobs() -> int:
    """Clear all pending jobs from the queue."""
    if not is_configured():
        return 0

    with _engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM backfill_jobs WHERE status = 'pending' RETURNING id"
        ))
        conn.commit()
        return len(result.fetchall())


def reorder_priority(job_id: int, new_priority: int) -> bool:
    """Change a job's priority."""
    if not is_configured():
        return False

    with _engine.connect() as conn:
        conn.execute(text("""
            UPDATE backfill_jobs
            SET priority = :priority
            WHERE id = :id AND status IN ('pending', 'paused')
        """), {'id': job_id, 'priority': new_priority})
        conn.commit()
    return True


def move_job_up(job_id: int) -> bool:
    """Move a job up in priority (decrease priority number). Atomic with row locking."""
    return _move_job(job_id, direction='up')


def move_job_down(job_id: int) -> bool:
    """Move a job down in priority (increase priority number). Atomic with row locking."""
    return _move_job(job_id, direction='down')


def _move_job(job_id: int, direction: str) -> bool:
    """
    Atomically move a job up or down in priority.

    Uses FOR UPDATE locking to prevent race conditions.
    Swaps priority with adjacent job in single transaction.
    """
    if not is_configured():
        return False

    with _engine.connect() as conn:
        # Lock and fetch the job we want to move
        job_result = conn.execute(text("""
            SELECT id, priority, created_at, status FROM backfill_jobs
            WHERE id = :id
            FOR UPDATE
        """), {'id': job_id})
        job = job_result.fetchone()

        if not job or job[3] not in ('pending', 'paused'):
            return False

        job_priority = job[1]
        job_created_at = job[2]

        # Find adjacent job based on direction (also lock it)
        if direction == 'up':
            adjacent_result = conn.execute(text("""
                SELECT id, priority FROM backfill_jobs
                WHERE status IN ('pending', 'paused')
                  AND (priority < :priority OR (priority = :priority AND created_at < :created_at))
                ORDER BY priority DESC, created_at DESC
                LIMIT 1
                FOR UPDATE
            """), {'priority': job_priority, 'created_at': job_created_at})
        else:  # down
            adjacent_result = conn.execute(text("""
                SELECT id, priority FROM backfill_jobs
                WHERE status IN ('pending', 'paused')
                  AND (priority > :priority OR (priority = :priority AND created_at > :created_at))
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                FOR UPDATE
            """), {'priority': job_priority, 'created_at': job_created_at})

        adjacent = adjacent_result.fetchone()

        if adjacent:
            adjacent_id = adjacent[0]
            adjacent_priority = adjacent[1]

            # Swap priorities atomically
            conn.execute(text("""
                UPDATE backfill_jobs SET priority = :new_priority WHERE id = :id
            """), {'id': job_id, 'new_priority': adjacent_priority})
            conn.execute(text("""
                UPDATE backfill_jobs SET priority = :new_priority WHERE id = :id
            """), {'id': adjacent_id, 'new_priority': job_priority})

            conn.commit()
            return True

        # No adjacent job found
        return False


def increment_retry(job_id: int) -> int:
    """Increment retry count and return new count."""
    if not is_configured():
        return 0

    with _engine.connect() as conn:
        result = conn.execute(text("""
            UPDATE backfill_jobs
            SET retry_count = retry_count + 1
            WHERE id = :id
            RETURNING retry_count
        """), {'id': job_id})
        conn.commit()
        row = result.fetchone()
        return row[0] if row else 0


def cleanup_old_jobs(days: int = 7) -> int:
    """Delete completed/failed jobs older than specified days."""
    if not is_configured():
        return 0

    # Validate days parameter
    if not isinstance(days, int) or days < 0:
        raise ValueError(f"days must be a non-negative integer, got {days}")

    with _engine.connect() as conn:
        # Use parameter binding with interval arithmetic (not string interpolation)
        result = conn.execute(text("""
            DELETE FROM backfill_jobs
            WHERE status IN ('completed', 'failed')
              AND completed_at < NOW() - (interval '1 day' * :days)
            RETURNING id
        """), {'days': days})
        conn.commit()
        return len(result.fetchall())


def get_running_job() -> Optional[Dict]:
    """Get the currently running job, if any."""
    if not is_configured():
        return None

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM backfill_jobs
            WHERE status = 'running'
            LIMIT 1
        """))
        row = result.fetchone()
        return dict(zip(result.keys(), row)) if row else None


def pause_running_job() -> bool:
    """Pause the currently running job."""
    running = get_running_job()
    if not running:
        return False
    return update_job_status(running['id'], 'paused')


def resume_paused_job(job_id: int) -> bool:
    """Resume a paused job by setting it to pending."""
    job = get_job(job_id)
    if not job or job['status'] != 'paused':
        return False
    return update_job_status(job_id, 'pending')


# ============================================================
# Gap Detection Helpers
# ============================================================

def get_price_date_coverage(symbol: str, start_date: date, end_date: date) -> List[date]:
    """Get list of dates that have price data for a symbol."""
    if not is_configured():
        return []

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date FROM raw_prices
            WHERE symbol = :symbol
              AND date >= :start_date
              AND date <= :end_date
            ORDER BY date
        """), {'symbol': symbol, 'start_date': start_date, 'end_date': end_date})
        return [row[0] for row in result.fetchall()]


def get_prediction_date_coverage(model_name: str, symbol: str, start_date: date, end_date: date) -> List[date]:
    """Get list of dates that have predictions for a model+symbol."""
    if not is_configured():
        return []

    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT run_time::date as date
            FROM predictions
            WHERE model_name = :model_name
              AND symbol = :symbol
              AND run_time >= :start_date
              AND run_time <= :end_date
            ORDER BY date
        """), {
            'model_name': model_name,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        return [row[0] for row in result.fetchall()]


def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """Generate list of trading days (weekdays) between dates."""
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days.append(current)
        current += timedelta(days=1)
    return days
