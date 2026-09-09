import os

from dotenv import load_dotenv # type: ignore
from groq import Groq # type: ignore

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


def main():
    print("Welcome to the AI Character Chat!")
    print("Type 'exit' to quit.")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
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

if __name__ == "__main__":
    main()