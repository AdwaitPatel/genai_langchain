from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from rich.markdown import Markdown
from rich.console import Console

console = Console()

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.8,
    google_api_key=os.getenv("GEMINI_API")
)

response = model.invoke("What is the capital of India")

md_response = response.content[0]['text']

console.print(Markdown(md_response))


