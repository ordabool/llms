from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Ollama(id="qwen3:8b"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Fetch stock price using the tool. Use tables to display data. Don't include any other text.",
    markdown=True,
)

# agent = Agent(
#     model=Gemini(id="gemini-2.5-pro"),
#     tools=[YFinanceTools(stock_price=True)],
#     instructions="Use the tool data exactly as returned. Do not make up or hallucinate any data. Display the exact price and timestamp from the tool response.",
#     markdown=True,
#     # debug_mode=True,
# )
agent.print_response("What is the stock price of Nvidia?", stream=True)