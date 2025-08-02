import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Ollama(
        id="llama3.2:3b",
        host=os.getenv("OLLAMA_HOST"),
        options={"temperature": 0.1},
    ),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Fetch stock price using the tool. Display only the price",
    markdown=True,
)

# agent = Agent(
#     model=Gemini(id="gemini-2.5-pro"),
#     tools=[YFinanceTools(stock_price=True)],
#     instructions="Use the tool data exactly as returned. Do not make up or hallucinate any data. Display the exact price and timestamp from the tool response.",
#     markdown=True,
#     # debug_mode=True,
# )
agent.print_response("What is the stock price of Apple?", stream=True)