from sqlalchemy import inspect, select

from database.connection import create_database_engine, create_schema, create_session_factory
from database.models import AgentRecord


def test_database_connects_creates_schema_and_round_trips_agent() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)

    assert {"agents", "tasks", "execution_runs", "execution_events"}.issubset(
        set(inspect(engine).get_table_names())
    )

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        agent = AgentRecord(name="Researcher", provider="openai", model="test-model")
        session.add(agent)
        session.commit()
        agent_id = agent.id

    with session_factory() as session:
        saved_agent = session.scalar(select(AgentRecord).where(AgentRecord.id == agent_id))
        assert saved_agent is not None
        assert saved_agent.name == "Researcher"
        assert saved_agent.provider == "openai"

    engine.dispose()