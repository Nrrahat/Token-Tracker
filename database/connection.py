"""SQLite engine and session construction."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config.settings import Settings, get_settings
from database.models import Base


def create_database_engine(database_url: str) -> Engine:
    """Create an engine and prepare the parent directory for file-backed SQLite."""
    url = make_url(database_url)
    connect_args: dict[str, object] = {}
    engine_options: dict[str, object] = {}

    if url.get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
        if url.database in (None, "", ":memory:"):
            engine_options["poolclass"] = StaticPool
        else:
            Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, connect_args=connect_args, **engine_options)

    if url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_schema(engine: Engine) -> None:
    """Create all currently defined tables in the configured database."""
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker:
    """Build a session factory bound to an engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database(settings: Settings | None = None) -> Engine:
    """Create the configured database schema and return its engine."""
    resolved_settings = settings or get_settings()
    engine = create_database_engine(resolved_settings.database_url)
    create_schema(engine)
    return engine