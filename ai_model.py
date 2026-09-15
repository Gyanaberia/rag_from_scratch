import os

from dotenv import load_dotenv # type: ignore
from groq import Groq #type: ignore
from vector_db import VectorDB
import streamlit as st  # type: ignore
load_dotenv()

class AIModel:
    db = None

    def __init__(self):
        self.db = VectorDB()
        self.db.process_knowledge_base

    def ai_model_api(self,messages):
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("AI_MODEL")

        groq = Groq(api_key=groq_api_key)
        chats = [
                {"role": m["role"], "content": m["content"]} 
                for m in messages
            ]
        
        chats[-1]["content"]=self.create_context_prompt(chats[-1]["content"])

        response = groq.chat.completions.create(messages=chats,
                                                model=groq_model,
                                                max_completion_tokens=1024,
                                                # reasoning_effort="medium",
                                                stop=None,
                                                stream=True)
        #for streaming responses, we need to iterate over the response object
        for chunk in response:
            if chunk.choices[0].delta.content is not None:#This prevents the initial reasoning message from being printed
                yield chunk.choices[0].delta.content


    def create_context_prompt(self, text):
        list_chunks=self.db.query_knowledge_base(text)
        context = ""
        i=1
        for chunk in list_chunks:
            context+="context phrase " + str(i)+":"+chunk+"\n"
            i+=1
        prompt_string = "Context:"+context+"\n User Query:"+text+"\n System Instruction: Create your response for the user query based solely in provided context and earlier conversation history. User is not aware of the context. Don't hint at context in the response. if the context doesn't contain the answer, say that you don't have information — don't guess."
        st.chat_message("ai_assistant").markdown(prompt_string)
        return prompt_string

    def chat_with_ai(self):
        # print("="*60)

        st.title("Welcome to AI chat!!")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Explore the world")
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.chat_message("user").markdown(user_prompt)
            ai_response = st.write_stream(self.ai_model_api(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    def query_db(self):
        print("="*60)
        print("Welcome to the Knowledge Base Query!")
        print(f"Provide your query followed by the number of top relevant chunks(chunk_count) you want to retrieve.The default count is %. Type 'exit' to quit.",{os.getenv("TOP_K")})
        while True:
            query = input("Enter your query: ")
            if query.lower() == 'exit':
                print("Exiting the knowledge base query.")
                print("="*60)
                break
            top_k = input("chunk_count: ")
            if not top_k.isdigit():
                top_k = int(os.getenv("TOP_K"))
            else:
                top_k = int(top_k)
            results = self.db.query_knowledge_base(query,top_k)
            print("Most relevant chunks from the knowledge base:")
            for i, chunk in enumerate(results):
                print(f"Chunk {i+1}: {chunk}")