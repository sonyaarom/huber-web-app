import os
import sys
from logging.config import fileConfig
from dotenv import load_dotenv

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

config = context.config


if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from hubert.db.models import Base
#----------------------------------------------------- #
target_metadata = Base.metadata

# Set the sqlalchemy.url from environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.venv')
load_dotenv(dotenv_path=dotenv_path)
db_url = os.getenv("DATABASE_URL") # Or build it from individual DB_ parts
if not db_url:
    # Try to build from individual parts
    db_user = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    if all([db_user, db_password, db_host, db_port, db_name]):
        if 'neon.tech' in db_host:
            endpoint_id = db_host.split('.')[0]
            db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require&options=endpoint%3D{endpoint_id}"
        else:
            db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

if not db_url:
    raise ValueError("DATABASE_URL environment variable not set, or DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME are not all set.")

config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() 