from sentence_transformers import SentenceTransformer

#from cold import extract
#from chunk import text_chunking

#text = extract("/Users/user/Documents/3A/Cold chain RAG exp/data/Thermal-Energy-Storage-Technologies.pdf")

#chunks = text_chunking(text, size=1000, overlap=150)

# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# embeddings = model.encode_document(
#     chunks,
#     normalize_embeddings=True, # vector normalize to unity
#     convert_to_numpy=True
# )

# query = "What domains of application use thermal energy storage?"

# query_embedding = model.encode_query(
#     query,
#     normalize_embeddings=True,
#     convert_to_numpy=True
# )

# print("Number of chunks:", len(chunks))
# print("Document embeddings shape:", embeddings.shape)
# print("Query embedding shape:", query_embedding.shape)

# print("First document vector:", embeddings[0][:10])
# print("Query vector:", query_embedding[:10])

def embed_documents(chunks):
    return model.encode_document(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


def embed_query(query):
    return model.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
