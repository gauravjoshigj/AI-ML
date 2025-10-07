# This is a conceptual example and may require specific libraries/APIs for actual implementation.
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import FunctionTool
# from file_writer import write_file
import os
from google.adk.tools.agent_tool import AgentTool



def write_file(filename: str, content: str, file_path: str = 'D:\Training\AI ML\Files_by_agent') -> str:
    """
    Writes content to a file, given the file path, filename, and content.
    Args:
        file_path (str): The directory path where the file should be saved.
        filename (str): The name of the file, including its extension (e.g., "my_document.txt").
        content (str): The string content to be written to the file.
    """
    os.makedirs(file_path, exist_ok=True)
    full_path = os.path.join(file_path, filename)
    try:
        with open(full_path, 'w',encoding='utf-8') as f:
            f.write(content)
        return f"File '{filename}' successfully written to '{file_path}'."
    except IOError as e:
        return f"Error writing file: {e}"

file_writer_tool = FunctionTool(func=write_file)

Search_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[GoogleSearchTool()]
)

# Create a single, master agent with both tools
root_agent = Agent(
    model="gemini-2.0-flash",
    name="Master_agent",
    description="A versatile agent that can perform Google searches and write information to a file.",
    instruction=(
        "You are a helpful assistant. Use the Google Search tool to find information "
        "and the write_file_tool to save information to a file when a user asks you to do so. "
        "For example, if a user asks you to 'write a summary of the latest news to a file named news.txt', "
        "you should first use the Google Search tool to find the latest news, then use the write_file_tool "
        "to save the summary to the specified file."
    ),
    tools=[AgentTool(agent=Search_agent),file_writer_tool],
)

