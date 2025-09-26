# agentics_mcp_news_sentiment.py
# Minimal Agentics + CrewAI script that calls your MCP server (mcp_yfinance_news.py)
# and has the agent analyze sentiment on the returned Yahoo Finance headlines.
#
# Requirements:
#   pip install agentics crewai crewai-tools mcp yfinance pandas
#
# LLM auth:
#   # Using Gemini **API key** (not Vertex):
#   export GEMINI_API_KEY="your_gemini_api_key_here"
#
# Run:
#   python agentics_mcp_news_sentiment.py AAPL 8

import sys, json
from pathlib import Path
from crewai import Agent, Task, Crew
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from agentics import AG

# ---------- inputs ----------
TICKER = "AAPL"
COUNT =  8

# Point to your MCP server file (the news server we built earlier)
SERVER_PATH = "/Users/boxuanli/Code/Agentics/mcp/finance/stock_recommendation/stock_news_mcp.py"

# Use the exact Python interpreter running this script + absolute path to server
params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])

# Open the MCP server as a tool collection
with MCPServerAdapter(params) as news_tools:
    # Choose an LLM via Agentics helper (avoids Vertex model confusion)
    # If your Agentics version expects different kwargs, this common form usually works:
    llm = AG.get_llm_provider("gemini")

    agent = Agent(
        role="Finance News Analyst",
        goal="Fetch Yahoo Finance news via MCP and produce a clear sentiment report as JSON.",
        backstory="You strictly call tools and return raw JSON with no extra commentary.",
        tools=news_tools,
        llm=llm,
        verbose=True,
    )

    task = Task(
        description=(
            f"Call the MCP tool `YFinanceNews` with ticker='{TICKER}', count={COUNT}. "
            "Then analyze the returned items and produce a sentiment report:\n"
            " - For each item: {title, link, publisher, published, score in [-1,1], "
            "   sentiment in ['positive','neutral','negative'], rationale}\n"
            " - Also compute overall_score in [-1,1] and a brief summary string.\n"
            "Return ONLY valid JSON in this exact shape:\n"
            "{\n"
            f'  "ticker": "{TICKER}",\n'
            f'  "count": {COUNT},\n'
            '  "per_item": [\n'
            '    {"title":"...", "link":"...", "publisher":"...", "published":"...", '
            '     "score": 0.0, "sentiment":"neutral", "rationale":"..."}\n'
            "  ],\n"
            '  "overall_score": 0.0,\n'
            '  "summary": "..." \n'
            "}\n"
            "Important: return JSON only, no markdown, no extra text."
        ),
        expected_output="A single JSON object exactly matching the schema.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()

    # Expecting the agent to return pure JSON string
    text = result.raw if hasattr(result, "raw") else str(result)
    data = json.loads(text)

    print(json.dumps(data, indent=2))
    out_path = f"{TICKER}_news_sentiment.json"
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"Saved to {out_path}")