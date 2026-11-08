import ollama

MODEL = "llama3.2:3b"  # llama3.1:8b

SYSTEM = (
    "Answer only from the passages provided and cite passage numbers like [17]. "
    "If the answer is not in the passages, say that you cannot find it."
)

def generate_answer(question, results):
    context = "\n\n---\n\n".join(f"[{r['id']}] {r['chunk']}" for r in results)
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]