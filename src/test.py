from embedding import embed_query
from V_store import FaissVectorStore
from llm import generate_answer

store = FaissVectorStore.load("data/V_store")

#query = "Does nuclear power plant use thermal energy storage?"
query = "What are the latent heat storage?"

results = store.search(embed_query(query), top_k=4)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} (id {result['id']}) ---")
    print("Similarity:", result["score"])
    print("Chunk:", result["chunk"][:300], "...")

print("\nAnswer")
print(generate_answer(query, results))