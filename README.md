1. Pdf extraction: raw text from the PDF with PyMuPDF
2. Text chunking 
    overlap is needed in case of the ending of this chunk is actually at the very begining of the next one so we overlap in order to not lose the context.
    - try implement first 
    - then can move on to use Langchain text splitter.
3. Embedding
    Use sentence-transformers/all-MiniLM-L6-v2: each chunk is turned into a vector so that texts with similar meaning end up close together.
    For retrieval tasks, for query embedding, encode_query() and encode_document() are the recommended methods. Many embedding models use different prompts or instructions for queries vs. documents, and these methods handle that automatically
4. FAISS (Facebook AI Similarity Search), a library designed to efficiently search for similar vectors
    Store and index the vector for faster searching
    We don't want to send all chunks to the LLM but to identify the 3 or 5 most relevant chunks.
    It indexes each chuck in a corresponding vector.
    
5. Semantic 
    So, how to get the 3 or 5 most relevant chucks? 
    Because all vectors are normalized, the inner product between the question vector and a chunk vector equals their cosine similarity, a score where higher means closer in meaning. FAISS computes this score against every chunk and returns the top k highest, along with their ids.

6. Generation


