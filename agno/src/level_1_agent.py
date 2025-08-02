import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools

# Load environment variables from .env file
load_dotenv()

# Test the YFinance tool directly
yfinance_tool = YFinanceTools(stock_price=True)
print("Testing YFinance tool directly:")
try:
    result = yfinance_tool.get_current_stock_price("AAPL")
    print(f"Direct tool result: {result}")
except Exception as e:
    print(f"Tool error: {e}")

print("\n" + "="*50 + "\n")

# agent = Agent(
#     model=Ollama(id="llama3.1:8b"),
#     tools=[YFinanceTools(stock_price=True)],
#     instructions="Use the tool data exactly as returned. Do not make up or hallucinate any data. Display the exact price and timestamp from the tool response.",
#     markdown=True,
# )

agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use the tool data exactly as returned. Do not make up or hallucinate any data. Display the exact price and timestamp from the tool response.",
    markdown=True,
)
agent.print_response("What is the stock price of Apple? Show the date and time of the data.", stream=True)