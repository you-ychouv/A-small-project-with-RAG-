
from sentence_transformers import SentenceTransformer
from cold import extract
from chunk import text_chunking
from V_store import FaissVectorStore

pdf_path = "/Users/user/Documents/3A/Cold chain RAG exp/data/Thermal-Energy-Storage-Technologies.pdf"

text = extract(pdf_path)
chunks = text_chunking(text, size=1000, overlap=150)

#print("Number of chunks:", len(chunks))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode_document(
    chunks,
    normalize_embeddings=True,
    convert_to_numpy=True
)

store = FaissVectorStore(dimension=embeddings.shape[1])

store.add_documents(chunks, embeddings)

store.save("data/V_store")

print("Database created!")
