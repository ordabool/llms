from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.ollama import OllamaEmbedder
from agno.models.google import Gemini

# Load Agno documentation in a knowledge base
# You can also use `https://docs.agno.com/llms-full.txt` for the full documentation
knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction.md"],
    vector_db=LanceDb(
        uri="tmp/lancedb_level_2",
        table_name="agno_docs",
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="dengcao/Qwen3-Embedding-8B:Q8_0"),
    ),
)

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/level_2_agent.db")

agent = Agent(
    name="Agno Assist",
    # model=Ollama(id="qwen3:8b"),
    model=Ollama(id="llama3.1:8b"),
    # model=Gemini(id="gemini-2.5-pro"),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    # Add the chat history to the messages
    add_history_to_messages=True,
    # Number of history runs
    num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    # Load the knowledge base, comment out after first run
    # Set recreate to True to recreate the knowledge base if needed
    agent.knowledge.load(recreate=False)
    agent.print_response("What are the difference between level 2 and level 3 agent? (in agno)", stream=True)
    