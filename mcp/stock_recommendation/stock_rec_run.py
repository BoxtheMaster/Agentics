# crew_ai_stock_recommend_demo.py
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from crewai import Agent, Task, Crew
from agentics.core.llm_connections import get_llm_provider
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import yaml

# ----- Output schema -----



### MAKE mcps-- one for yf and one for news (DDG/yf/wrds, etc)
class StockReco(BaseModel):
    ticker: str
    company: Optional[str] = None
    recommendation: str
    rationale: str
    sentiment_overall: float = Field(..., description="Overall sentiment score in [-1,1]")
    price_last_close: Optional[float] = None
    price_ret_15: Optional[float] = None

params = StdioServerParameters(
    command="python",
    args=["/Users/boxuanli/Documents/Code/agentics/mcp/stock_recommend.py"],
)

with MCPServerAdapter(params) as stock_tools:
    doc_agent = Agent(
        role="Stock analyst",
        goal="Call MCP tool 'stock_recommend' to produce a concise structured recommendation.",
        backstory="Analyzes a single ticker by invoking MCP tools.",
        tools=stock_tools,
        llm=get_llm_provider("gemini"),
        verbose=True,
    )

    doc_task = Task(
        description=(
            "Ticker: {{ticker}}.\n"
            "Invoke MCP tool `stock_recommend` with:\n"
            "  - ticker='{{ticker}}'\n"
            "  - days=7\n"
            "  - max_results=8\n"
            "  - risk='medium'\n"
            "Return only the structured fields mapped to the StockReco schema.\n"
            "Set price_last_close from price_snapshot.last_close and price_ret_15 from price_snapshot.ret_15."
        ),
        expected_output="A single JSON object following the StockReco schema.",
        agent=doc_agent,
        output_pydantic=StockReco,
    )

    crew = Crew(agents=[doc_agent], tasks=[doc_task], verbose=True)
    result = crew.kickoff(inputs={"ticker": "AAPL"})

    if result.pydantic:
        print(yaml.dump(result.pydantic.model_dump(), sort_keys=False))