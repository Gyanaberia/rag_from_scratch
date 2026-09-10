import os

from dotenv import load_dotenv # type: ignore
from groq import Groq #type: ignore
from vector_db import VectorDB # type: ignore

load_dotenv()

def ai_char(message):
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("AI_MODEL")

    groq = Groq(api_key=groq_api_key)
    response = groq.chat.completions.create(messages=message,
                                            model=groq_model,
                                            max_completion_tokens=1024,
                                            # reasoning_effort="medium",
                                            stop=None,
                                            stream=True)
    ai_response = ""
    #for streaming responses, we need to iterate over the response object
    for chunk in response:
         if chunk.choices[0].delta.content is not None:#This prevents the initial reasoning message from being printed
            print(chunk.choices[0].delta.content, end="")
            ai_response += chunk.choices[0].delta.content
    return ai_response

def chat_with_ai():
    print("="*60)
    print("Welcome to the AI Chat!")
    print("Type System to set a system message. This helps provide more context or instructions to the AI.")
    print("Type 'exit' to quit.")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            print("="*60)
            break
        # Allow the user to set a system message. This helps provide more context or instructions to the AI character.
        if user_input.lower() == 'system':
            system_message = input("Enter system message: ")
            messages.append({"role": "system", "content": system_message})
            print("System message set.")
            continue
        #Keep appending to the messages list to maintain context for the AI character. This allows the AI to remember previous interactions and respond more naturally.
        messages.append({"role": "user", "content": user_input})
        response = ai_char(messages)
        print("\n")
        messages.append({"role": "assistant", "content": response})

def query_db(db):
    print("="*60)
    print("Welcome to the Knowledge Base Query!")
    print("Provide your query followed by the number of top relevant chunks(chunk_count) you want to retrieve.The default count is 1. Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            print("Exiting the knowledge base query.")
            print("="*60)
            break
        top_k = input("chunk_count: ")
        if not top_k.isdigit():
            top_k = 1
        else:
            top_k = int(top_k)
        results = db.query_knowledge_base(query,top_k)
        print("Most relevant chunks from the knowledge base:")
        for i, chunk in enumerate(results):
            print(f"Chunk {i+1}: {chunk}")

def main():
    print("Welcome. Type 'chat to start chatting with the AI")
    print("Type 'query' to query the knowledge base. This will return the most relevant chunks from the knowledge base for your query.")
    print("Type 'exit' to quit the program.")
    while True:
        user_input = input("Enter your choice (chat/query/exit): ")
        if user_input.lower() == 'query':
            db = VectorDB()
            db.process_knowledge_base()
            query_db(db)
        elif user_input.lower() == 'chat':
            chat_with_ai()
        else:
            print("Exiting the program.")
            break

if __name__ == "__main__":
    main()


