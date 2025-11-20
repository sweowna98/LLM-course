# Install required packages
"""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate torch datasets huggingface_hub
pip install pandas bitsandbytes sentence-transformers faiss-cpu langchain chromadb
pip install --no-cache-dir numpy==1.26.4
pip install gradio
pip install scikit-learn evaluate
pip install requests 
"""

# Imports
import model
import data_mining


from sentence_transformers import SentenceTransformer

# Retrieval function and similarity comparison

def retrieve(query, top_k, threshold):
    query_emb = data_mining.embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = data_mining.index.search(query_emb, top_k)

    # Convert L2 to pseudo-similarity
    similarities = 1 / (1 + distances[0])

    # Compute max similarity
    max_similarity = float(similarities.max())

    # Only include rows above threshold
    relevant_texts = [data_mining.texts[i] for i, sim in zip(indices[0], similarities) if sim >= threshold]

    # Return only if threshold satisfied
    return relevant_texts, max_similarity

# Generation based on RAG similarity

def generate_answer(hist, query, top_k=5, threshold=0.43):
    context_rows, max_similarity = retrieve(query, top_k, threshold)

    if context_rows and max_similarity >= threshold:
        print(f"Using RAG (similarity={max_similarity:.2f} ≥ {threshold})")
        #  RAG prompt
        context = "\n".join(context_rows)
        prompt = f"""
        Instructions:
        You are a friendly and knowledgeable travel assistant that helps users plan short trips and vacations. 
        The "Context" section below includes a dataset of real travelers, their preferences, destinations, and trip details. 
        Use this data as a general guide to infer travel patterns and make recommendations.
        Do NOT make bullet points.
        When suggesting locations or activities:
        - Prefer destinations that are geographically close to the traveler’s starting location, unless they explicitly request otherwise.
        - Keep your responses brief (1–3 sentences maximum).
        - Speak conversationally — no lists or bullet points.
        - Use confident, friendly language.
        - If the dataset doesn’t provide relevant information, use your general travel knowledge to give useful advice.
        - Only mention gender or age of the user if they have already been mentioned.
        - Try to ask for prices/cost of the user and match it with the Context.
        - Do NOT mention the user previous travels and/or preferences.

        Always maintain privacy:
        - Never mention “the dataset” explicitly in your reply.
        - You may generalize patterns (e.g., “many travelers from Germany enjoy city breaks in Italy”), but keep it natural.

        If the user’s query is ambiguous, ask one short clarifying question instead of making assumptions.

        Context:
        {context}

        Chat History: {hist}

        User: {query}
        Assistant: """
        
    else:
        print(f"Skipping RAG (similarity={max_similarity:.2f} < {threshold})")
        prompt = f"""
        Instructions:
        You are a friendly travel assistant. Only provide travel suggestions if the user asks a question about a trip. 
        If the user just says 'Hi', 'Hello', or a greeting, respond with a friendly greeting and a short clarifying question about their travel preferences. 
        If the user does not specify his location and travel needs, ask for further clarification.
        Do NOT invent travel suggestions for greetings.
        
        Chat History: {hist}

        User: {query}
        Assistant: """

    output = llm(prompt)[0]["generated_text"]
    print(hist)
    out = output.split("Assistant:") # Split on assistant to keep what we need
    return out[-1]



