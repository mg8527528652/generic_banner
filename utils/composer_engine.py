import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

