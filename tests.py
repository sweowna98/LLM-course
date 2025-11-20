import gradio as gr
import time
import statistics
import sys

# Simple Geo Tests 
hist = ""

queries = [
    "Hi!",
    "I'm a 41 year old female traveler living in Great Britain. I want to go for about a week that's not far away, what would fit?",
    "I'm a 41 year old female traveler living in Germany. I want to go for about a week that's not far away, what would fit?",
    "Im 18 and want to have fun someplace warm!, im looking for someplace cheap what can you give me",
    "I'm a 41 year old female traveler living in China. I want to go for about a week that's not far away, what would fit?",
    "I'm a 41 year old female traveler living in Moscow. I want to go for about a week that's not far away, what would fit?"
]

for q in queries:
    print("Q:", q)
    answer = generate_answer(hist, q)
    hist = ""
    print("A:", answer)
    print("------")




# Chat test
def chatWindow(query, history):
    
    hist = ""
    for x in history:
        hist = hist + x['role'].title() + ": " + x['content'] + "\n"
        

    answer = generate_answer(hist, query)
    #hist = hist + "\nUser: " + query + "\nAssistant: " + answer

    return answer

demo = gr.ChatInterface(fn=chatWindow, type="messages", title="Travel Bot")
demo.launch()




# Tests Quantitative
example_texts = ["Hi!", "I'm a 41 year old female traveler living in China. I want to go for about a week that's not far away, what would fit?", "I want to go somewhere warm in indonesia, what would fit?",
                "What are some interesting cities in Great Britain/UK?", "Is Barcalona a good place for a young person to visit, what can I do there?", "Hi!", "I'm a 41 year old female traveler living in China. I want to go for about a week that's not far away, what would fit?", "I want to go somewhere warm in indonesia, what would fit?",
                "What are some interesting cities in Great Britain/UK?", "Is Barcalona a good place for a young person to visit, what can I do there?", "Im Theodor!", "Whats close countries to visit, im in Japan",
                "How is indonesia for beach holiday", "what are some good cities in eastern europe", "I live in Moscow, where can i go for culture travel thats close?"]
results_rag = []
results_mistral = []
hist = ""
# RAG
for example in example_texts:
    start_time = time.time()
    context_rows, max_similarity = retrieve("Hi!", 5, 0.42)
    end_time = time.time()
    results_rag.append(end_time - start_time)

print(results_rag)
print("Average of RAG calls: " + str(sum(results_rag)/len(results_rag)))
print("Range:", max(results_rag) - min(results_rag))
print("Variance:", statistics.variance(results_rag))
print("Standard Deviation:", statistics.stdev(results_rag))

# Mistal/LLM
for example in example_texts:
    start_time = time.time()
    answer = generate_answer(hist, example)
    hist = ""
    end_time = time.time()
    results_mistral.append(end_time - start_time)

print(results_mistral)
print("Average of Mistral calls: " + str(sum(results_mistral)/len(results_mistral)))
print("Range:", max(results_mistral) - min(results_mistral))
print("Variance:", statistics.variance(results_mistral))
print("Standard Deviation:", statistics.stdev(results_mistral))
