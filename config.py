import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
COLLECTION_NAME="HARRY POTTER COLLECTION"
PDF_PATH="reports/harrypotter.pdf"

