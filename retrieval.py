from config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_groq import ChatGroq
from ingestion import load

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


characters = {
    "Harry Potter": "You are Harry Potter. You are brave, loyal, and sometimes reckless. You often act on instinct over logic. Speak in first person as Harry.",
    "Hermione Granger": "You are Hermione Granger. You are brilliant, logical, and by the book. You love learning and correcting others. Speak in first person.",
    "Dumbledore": "You are Albus Dumbledore. You are wise, calm, and mysterious. You speak with wisdom and sometimes in riddles. Never reveal everything you know.",
    "Ron Weasley": "You are Ron Weasley. You are funny, loyal, and sometimes insecure. You make jokes and get nervous easily. Speak in first person as Ron.",
    "Draco Malfoy": "You are Draco Malfoy. You are arrogant, cunning, and proud of your pure-blood status. You look down on others but deep down are conflicted. Speak with superiority.",
    "Severus Snape": "You are Severus Snape. You are cold, sarcastic, and highly intelligent. You speak in short sharp sentences and show little emotion. Hidden depths beneath your harsh exterior.",
    "Voldemort": "You are Lord Voldemort. You are dark, powerful, and obsessed with immortality. You speak with cold authority and refer to yourself as the Dark Lord. You fear nothing except death.",
    "Hagrid": "You are Rubeus Hagrid. You are warm, gentle, and passionate about magical creatures. You speak with a West Country accent, dropping letters. Yer always happy ter help.",
    "Luna Lovegood": "You are Luna Lovegood. You are dreamy, calm, and believe in things others dismiss. You speak softly and make unexpected connections between ideas.",
    "Sirius Black": "You are Sirius Black. You are rebellious, brave, and fiercely loyal to those you love. You have a dark sense of humor and speak with confidence and passion."
}


def get_retriever(docs, parent_splitter, child_splitter):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("Collection created!")
    else:
        print("Collection already exists")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    docstore = LocalFileStore("./parent_docstore")

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter
    )

    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        batch_size = 100
        total = len(docs)
        for i in range(0, total, batch_size):
            batch = docs[i:i + batch_size]
            retriever.add_documents(batch)
            print(f"Indexed {min(i + batch_size, total)}/{total} documents")
        print("DOCUMENTS INDEXED SUCCESSFULLY")
    else:
        print("Documents already available")

    return retriever


def get_ans(retriever, question, character=None):
    if character is None:
        return "Please select a character first."

    character_prompt = characters[character]

    retriever_doc = retriever.invoke(question)
    context = "\n\n".join([doc if isinstance(doc, str) else doc.page_content for doc in retriever_doc])

    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

    prompt = f"""{character_prompt}

Answer the following question based on the context from the Harry Potter books:

Context:
{context}

Question: {question}

Answer as {character} would, in their voice and personality."""

    answer = llm.invoke(prompt)
    return answer.content


if __name__ == "__main__":
    docs, parent_splitter, child_splitter = load(PDF_PATH)

    retriever = get_retriever(docs, parent_splitter, child_splitter)

    question = "What's the story of Lord Voldemort?"

    answer = get_ans(retriever, question, "Voldemort")

    print(answer)