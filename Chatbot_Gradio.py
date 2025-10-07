import gradio as gr
import os
import logging
import json
from dotenv import load_dotenv
import time
import requests

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
# Set up Google Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                  

# Function to call Google Gemini API along with history
def call_gemini_api(prompt, history):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Format history for Gemini API
    gemini_contents = []
    for user_text, bot_text in history:
        gemini_contents.append({"role": "user", "parts": [{"text": user_text}]})
        gemini_contents.append({"role": "model", "parts": [{"text": bot_text}]})
    gemini_contents.append({"role": "user", "parts": [{"text": prompt}]})  # Add the current prompt

    data = {
        "contents": gemini_contents,
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.7,
            "topP": 0.95,
            "topK": 40
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        logging.error(f"Gemini API Error: {response.status_code} - {response.text}")
        return "Error: Unable to get response from Gemini API."
 

#generate Gradio UI which allows user to input a question and get an answer from Google Gemini API
def chatbot_response(message, history):
    gemini_response = call_gemini_api(message, history)
    print(gemini_response)    
    history.append((message, gemini_response))
    return gemini_response  

with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        fn=chatbot_response, 
        title="My Simple Chatbot",
        description="Type a message to chat with the bot."
    )

demo.launch()
