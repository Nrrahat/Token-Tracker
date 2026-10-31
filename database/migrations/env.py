from alembic import context
from sqlalchemy import engine_from_config, pool

from config.constants import DEFAULT_DATABASE_URL
from config.settings import get_settings
from database.models import Base

config = context.config
configured_url = config.get_main_option("sqlalchemy.url")
settings_url = get_settings().database_url
if configured_url == DEFAULT_DATABASE_URL and settings_url != DEFAULT_DATABASE_URL:
    config.set_main_option("sqlalchemy.url", settings_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()