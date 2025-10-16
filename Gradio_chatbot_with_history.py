import os
import gradio as gr
from google import genai
from google.genai import types

# --- Configuration ---
# Your API key must be available as an environment variable (e.g., GEMINI_API_KEY)
# The client automatically uses the environment variable.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    print("Please make sure your GEMINI_API_KEY is set correctly.")
    client = None

# Specify the model you want to use
MODEL_NAME = "gemini-2.5-flash"

# --- Gemini API Call Logic ---

def gradio_history_to_gemini_contents(history: list) -> list:
    """
    Converts Gradio's chat history format (list of dicts) into 
    the list of Content objects required by the Gemini API's 
    generate_content method for conversational turns.

    The Gradio format for history (when type="messages") is:
    [
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "..."}
        ...
    ]
    """
    gemini_contents = []
    
    # Gradio history contains all previous turns, including the last user message
    for message in history:
        role = message.get("role")
        text = message.get("content")
        
        # Map Gradio roles to Gemini roles
        gemini_role = "user" if role == "user" else "model"
        
        # Create a Content object
        content = types.Content(
            role=gemini_role,
            parts=[types.Part.from_text(text)]
        )
        gemini_contents.append(content)
        
    return gemini_contents

def gemini_chatbot(message: str, history: list) -> str:
    """
    The main function called by gr.ChatInterface.
    It takes the new user message and the entire conversation history.
    """
    if not client:
        return "Gemini Client not initialized. Check your API key."

    # 1. Gradio automatically adds the current user 'message' to 'history' 
    # before calling this function, but only if 'type="messages"' is used, 
    # and the history passed to the function includes *only* the preceding turns.
    # Therefore, we manually reconstruct the full conversation.
    
    # The history passed to the function is typically *everything before* the 
    # current user message. The current 'message' is the latest one.
    
    # Reconstruct the full conversation history including the latest user message
    full_history = history + [{"role": "user", "content": message}]
    
    # 2. Convert the Gradio history format to the Gemini API format (list of Content objects)
    gemini_contents = gradio_history_to_gemini_contents(full_history)

    try:
        # 3. Call the generate_content API with the full conversation history
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=gemini_contents,
            # For streaming, use generate_content_stream and yield chunks
            stream=False 
        )
        
        # 4. Return the generated text response
        return response.text

    except Exception as e:
        return f"An error occurred with the Gemini API call: {e}"

# --- Gradio Interface ---

if client:
    demo = gr.ChatInterface(
        fn=gemini_chatbot,
        title=f"Gemini {MODEL_NAME} Chatbot",
        description="Ask me anything! The conversation history is maintained via the ChatInterface.",
        # 'type="messages"' is important, as it passes history in the 
        # role/content dictionary format expected by our function.
        # This format is also required for streaming and multimodal support.
        type="messages", 
        examples=[
            "What is the capital of France?",
            "What is a large language model?",
            "Write a short, funny poem about a cat."
        ]
    )

    if __name__ == "__main__":
        demo.launch()
else:
    # Handle the case where the client failed to initialize
    print("Gradio ChatInterface will not launch due to missing API key.")
