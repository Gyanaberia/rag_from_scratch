from ai_model import AIModel
import streamlit as st # type: ignore

@st.cache_resource
def get_ai_bot():
    return AIModel()

def main():
    ai_bot = get_ai_bot()
    ai_bot.chat_with_ai()

if __name__ == "__main__":
    main()
