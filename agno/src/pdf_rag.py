from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.storage.sqlite import SqliteStorage
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder

knowledge = PDFKnowledgeBase(
    path="pdfs/Dragonbane",
    vector_db=PgVector(
        table_name="dragonbane_documents",
        db_url="postgresql+psycopg://ai:ai@pgvector:5432/ai",
        # embedder=OllamaEmbedder(id="dengcao/Qwen3-Embedding-8B:Q8_0", dimensions=4096), # Works well with qwen3
        embedder=OllamaEmbedder(id="mxbai-embed-large:335m", dimensions=1024), # Works well with llama3.1
    ),
    reader=PDFReader(chunk=True),
)

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/dragonbane_agent.db")

agent = Agent(
    name="Dragonbane Assistant",
    # model=Ollama(id="qwen3:8b"),
    model=Ollama(id="llama3.1:8b"),
    instructions=[
        "You are a helpful assistant that can answer questions about the Dragonbane TTRPG game.",
        "Search your knowledge before answering the question. It holds all of the rules for the game.",
        "Give priority to the rules PDF over other PDFs.",
        "State the page number of the rule in your response.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    # add_history_to_messages=True,
    # num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    # Load the knowledge base, comment out after first run
    # Set recreate to True to recreate the knowledge base if needed
    agent.knowledge.load(recreate=False)
    agent.print_response("Give an overview of the Dragonbane playable kins.", stream=True)
    