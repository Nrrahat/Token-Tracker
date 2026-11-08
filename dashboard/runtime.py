"""Construct the application's shared runtime services."""

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from agents.agent_manager import AgentManager
from agents.execution_loop import AgentExecutor
from analytics.service import AnalyticsService
from config.settings import get_settings
from database.connection import create_database_engine, create_session_factory
from instrumentation.event_tracker import EventTracker
from instrumentation.llm_tracker import LLMTracker
from instrumentation.trace_manager import TraceManager
from evaluation.service import EvaluationManager
from experiments.comparison_engine import ComparisonEngine
from experiments.dataset_manager import DatasetManager
from experiments.experiment_manager import ExperimentManager
from experiments.experiment_runner import ExperimentRunner
from skills.skill_loader import SkillLoader
from skills.skill_registry import SkillRegistry
from token_engine.tokenizer import TiktokenTokenizer
from tools.available.calculator import CalculatorTool
from tools.available.safe_document_reader import SafeDocumentReaderTool
from tools.custom_python_tool import sync_custom_tools
from tools.custom_tool_store import CustomToolStore
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolRegistry


@dataclass
class AppServices:
    engine: Engine
    session_factory: sessionmaker
    uploaded_root: Path
    agent_manager: AgentManager
    skill_registry: SkillRegistry
    tool_registry: ToolRegistry
    tool_manager: ToolManager
    custom_tool_store: CustomToolStore
    event_tracker: EventTracker
    llm_tracker: LLMTracker
    trace_manager: TraceManager
    agent_executor: AgentExecutor
    analytics_service: AnalyticsService
    evaluation_manager: EvaluationManager
    experiment_manager: ExperimentManager
    experiment_runner: ExperimentRunner
    comparison_engine: ComparisonEngine


def build_services() -> AppServices:
    """Apply database migrations and construct shared business services."""
    project_root = Path(__file__).resolve().parents[1]
    settings = get_settings()
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.upgrade(alembic_config, "head")

    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    upload_root = project_root / "storage" / "uploaded_files"
    upload_root.mkdir(parents=True, exist_ok=True)

    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(SafeDocumentReaderTool(upload_root))
    tool_manager = ToolManager(tool_registry, tokenizer=tokenizer)
    custom_tool_store = CustomToolStore(project_root / "custom_tools")
    sync_custom_tools(custom_tool_store, tool_registry)

    skill_registry = SkillRegistry(
        project_root / "skills" / "available",
        loader=SkillLoader(tokenizer=tokenizer),
    )
    agent_manager = AgentManager(session_factory, skill_registry, tool_registry)
    event_tracker = EventTracker(session_factory)
    llm_tracker = LLMTracker(session_factory, event_tracker=event_tracker)
    trace_manager = TraceManager(session_factory)
    agent_executor = AgentExecutor(
        session_factory,
        agent_manager,
        skill_registry,
        tool_registry,
        tool_manager,
        llm_tracker,
        event_tracker,
    )
    analytics_service = AnalyticsService(session_factory)
    evaluation_manager = EvaluationManager(session_factory)
    experiment_manager = ExperimentManager(
        session_factory,
        agent_manager,
        dataset_manager=DatasetManager(),
    )
    comparison_engine = ComparisonEngine(session_factory, experiment_manager)
    experiment_runner = ExperimentRunner(
        experiment_manager,
        agent_manager,
        agent_executor,
        evaluation_manager,
        comparison_engine,
    )
    return AppServices(
        engine=engine,
        session_factory=session_factory,
        uploaded_root=upload_root,
        agent_manager=agent_manager,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        tool_manager=tool_manager,
        custom_tool_store=custom_tool_store,
        event_tracker=event_tracker,
        llm_tracker=llm_tracker,
        trace_manager=trace_manager,
        agent_executor=agent_executor,
        analytics_service=analytics_service,
        evaluation_manager=evaluation_manager,
        experiment_manager=experiment_manager,
        experiment_runner=experiment_runner,
        comparison_engine=comparison_engine,
    )