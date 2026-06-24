"""LangGraph ReAct agent wiring the LLM, tools, and privacy guard together.

The agent can (a) answer research questions from the indexed corpus and
(b) call the live forecasting API — choosing tools autonomously. Every final
answer is passed through :class:`~rag_agent.privacy.PrivacyGuard` before it
leaves the process.

LangChain / LangGraph / Anthropic imports are lazy so the rest of the package
(and its tests) need not install the full agent stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, api_key, load_config
from .exceptions import RagAgentError
from .logging_config import configure_logging, get_logger
from .privacy import PrivacyGuard
from .tools import forecast_demand, retrieve_research

log = get_logger(__name__)


@dataclass
class RagAgent:
    """A compiled agent plus the runtime collaborators it needs."""

    graph: Any
    cfg: Config
    guard: PrivacyGuard

    def answer(self, question: str) -> str:
        """Answer ``question``, returning a privacy-filtered final response."""
        result = self.graph.invoke({"messages": [("user", question)]})
        messages = result.get("messages", [])
        if not messages:
            raise RagAgentError("Agent returned no messages.")
        content = getattr(messages[-1], "content", "")
        text = content if isinstance(content, str) else str(content)
        return self.guard.filter_output(text)


def _build_tools(cfg: Config, collection: Any, guard: PrivacyGuard) -> list[Any]:
    """Adapt the pure tool functions into LangChain StructuredTools."""
    from langchain_core.tools import StructuredTool

    def research_tool(question: str) -> str:
        """Search Md Minhazur Rahman's research corpus for relevant passages."""
        return retrieve_research(question, cfg, collection, guard)

    def forecast_tool(rows: list[dict[str, float]]) -> list[float]:
        """Forecast demand for synthetic feature rows via the forecasting API."""
        return forecast_demand(rows, cfg, guard)

    return [
        StructuredTool.from_function(
            research_tool,
            name="retrieve_research",
            description=(
                "Retrieve relevant passages from the research corpus "
                "(dissertation + preprint). Input: a natural-language question."
            ),
        ),
        StructuredTool.from_function(
            forecast_tool,
            name="forecast_demand",
            description=(
                "Run live demand forecasts. Input: 'rows', a list of synthetic "
                "pre-engineered feature dicts. Returns predicted demand values."
            ),
        ),
    ]


def build_agent(cfg: Config | None = None) -> RagAgent:
    """Construct the full agent from configuration.

    Raises:
        ConfigError: If ``ANTHROPIC_API_KEY`` is missing.
        RagAgentError: If the agent stack cannot be initialised.
    """
    cfg = cfg or load_config()
    guard = PrivacyGuard(
        fail_closed=cfg.privacy.fail_closed,
        max_output_chars=cfg.privacy.max_output_chars,
    )
    try:
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent

        from .vectorstore import get_collection
    except ImportError as exc:  # pragma: no cover - env guard
        raise RagAgentError("Agent dependencies missing; install requirements.txt.") from exc

    llm = ChatAnthropic(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        api_key=api_key(),
    )
    collection = get_collection(cfg)
    tools = _build_tools(cfg, collection, guard)
    graph = create_react_agent(llm, tools, state_modifier=cfg.agent.system_prompt)
    log.info("agent_built", model=cfg.llm.model, tools=len(tools))
    return RagAgent(graph=graph, cfg=cfg, guard=guard)


def main() -> None:
    """Tiny CLI for a one-shot question (useful for manual smoke checks)."""
    import argparse

    parser = argparse.ArgumentParser(description="Ask the RAG agent a question.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    print(build_agent(cfg).answer(args.question))


if __name__ == "__main__":
    main()
