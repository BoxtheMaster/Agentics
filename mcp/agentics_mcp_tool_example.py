import asyncio
import os
from typing import Optional

from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentics import AG
from agentics.core.llm_connections import available_llms
from mcp import StdioServerParameters  # For Stdio Server

load_dotenv()





# class SearchResult(BaseModel):
#     title: Optional[str]
#     text: Optional[str]
#     source_url: Optional[str]


# class WebSearchReport(BaseModel):
#     """Detailed Report of market news related to a stock"""
#     detailed_report: Optional[str] = Field(
#         None,
#         description="Markdown document containing relevant background information for the question being answered.",
#     )
#     relevant_references: list[SearchResult] = Field(
#         [],
#         description="Relevant Snippets of text extracted from web search that supports the answers. Do not make up text, just copy from search results if relevant",
#     )

class NewsSentimentReport(BaseModel):
    ticker: Optional[str]=None
    relevant_news: Optional[list[str]]=None
    price_close: Optional[float]=None
    price_open: Optional[float]=None
    price_prev_close: Optional[float]=None
    year__high: Optional[float]=None
    year_low: Optional[float]=None
    Volume: Optional[float]=None
    sentiment_score: Optional[float]=0
    sentiment_report_summary: Optional[str]=None
    investment_recommendation: Optional[str]=None




import streamlit as st
st.header("Agentic MCP demo")
# mcp_map={"stock_news": os.getenv("NEWS_MCP_TOOL_EXAMPLE_PATH"),
#          "stock_price": os.getenv("STOCK_MCP_TOOL_EXAMPLE_PATH")}
question=st.text_input("Ask a question")
select_tools=st.multiselect("Select your tool", options=['stock_news','stock_price'])
if question:
    news_tool = StdioServerParameters(
        command="python3",
        args=[os.getenv("NEWS_MCP_TOOL_EXAMPLE_PATH")],
        env={"UV_PYTHON": "3.12", **os.environ},
    )

    price_tool= StdioServerParameters(
        command="python3",
        args=[os.getenv("STOCK_MCP_TOOL_EXAMPLE_PATH")],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    if select_tools==['stock_news']:
        with MCPServerAdapter(news_tool) as server_tools:
            print(
                f"Available tools from Stdio MCP server: {[tool.name for tool in server_tools]}"
            )

            results = asyncio.run(
                AG(
                    atype=NewsSentimentReport,
                    tools=server_tools,
                    max_iter=10,
                    verbose_agent=True,
                    reasoning=True,
                    #description="Extract stock market news for the input day",
                    llm=AG.get_llm_provider("gemini"),
                )
                << [question]
            )
        st.write(results[0])
    if select_tools==['stock_price']:
        with MCPServerAdapter(price_tool) as server_tools:
            print(
                f"Available tools from Stdio MCP server: {[tool.name for tool in server_tools]}"
            )

            results = asyncio.run(
                AG(
                    atype=NewsSentimentReport,
                    tools=server_tools,
                    max_iter=10,
                    verbose_agent=True,
                    reasoning=True,
                    description="Extract stock market price, volume, day high, low for the input day and give investment decision",
                    llm=AG.get_llm_provider("gemini"),
                )
                << [question]
            )
        st.write(results[0])
    if select_tools==['stock_news','stock_price'] or select_tools==['stock_price', 'stock_news']:
        with (MCPServerAdapter(price_tool) as price_tools,
             MCPServerAdapter(news_tool) as news_tools,
             ):
            server_tools=news_tools+price_tools
            
            print(
            f"Available tools from Stdio MCP server: {[tool.name for tool in price_tools]}"
            )
            print(
        f"Available tools from Stdio MCP server: {[tool.name for tool in news_tools]}"
            )

            results = asyncio.run(
                AG(
                    atype=NewsSentimentReport,
                    role="Stock adivse Agent",
                    goal="Give stock recommendation to user using the available MCP tool.",
                    tools=server_tools,
                    max_iter=10,
                    verbose_agent=True,
                    reasoning=True,
                    description="Extract stock market news and do sentiment analysis, and also consider price and volume for the input day and give investment recommendation",
                    llm=AG.get_llm_provider("gemini"),
                )
                << [question]
            )
        st.write(results[0])
        
    
        #print(results.pretty_print())



 