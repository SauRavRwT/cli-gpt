from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# Initialize Rich console for better formatting
console = Console()

# Load environment variables from .env file
load_dotenv()

# Create necessary directories
HISTORY_DIR = Path("./saved")
HISTORY_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR = HISTORY_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Set up API keys
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# Available models and their configurations
MODELS = {
    "gemini": {
        "name": "gemini-2.0-flash",
        "provider": "gemini"
    },
    "llama": {
        "name": "llama-3.3-70b-versatile",
        "provider": "groq"
    }
}

# Initialize clients
clients = {
    "gemini": OpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        default_headers={"x-goog-api-key": gemini_api_key}
    ),
    "groq": OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
}

def format_response(text):
    """Format the response with Markdown support"""
    try:
        return Panel(Markdown(text), border_style="cyan", padding=(1, 2))
    except Exception:
        return Panel(text, border_style="cyan", padding=(1, 2))

class Conversation:
    def __init__(self, model_name: str):
        self.messages = [
            {
                "role": "system",
                "content": """You are a highly knowledgeable and helpful CLI assistant. 
                - Provide clear, concise, and accurate responses
                - Use Markdown formatting when beneficial
                - Include code examples when relevant
                - Be friendly but professional"""
            }
        ]
        self.model_name = MODELS[model_name]["name"]
        self.provider = MODELS[model_name]["provider"]
        self.start_time = datetime.now()
    
    def add_message(self, role: str, content: str):
        if content is not None:
            self.messages.append({"role": role, "content": content})
    
    def save(self):
        filename = f"{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(CONVERSATIONS_DIR / filename, "w") as f:
            json.dump({
                "model": self.model_name,
                "timestamp": self.start_time.isoformat(),
                "messages": self.messages
            }, f, indent=2)

def ask_gpt(prompt: str, conversation: Conversation) -> str:
    try:
        conversation.add_message("user", prompt)
        
        with Progress() as progress:
            task = progress.add_task("[yellow]Thinking...", total=None)
            client = clients[conversation.provider]
            
            # Convert messages to the format expected by the API
            messages: List[ChatCompletionMessageParam] = []
            for msg in conversation.messages:
                if msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                elif msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            response = client.chat.completions.create(
                model=conversation.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No response content received from the model")
            
        reply = content.strip()
        conversation.add_message("assistant", reply)
        return reply
    
    except Exception as e:
        error_msg = str(e)
        if "model_decommissioned" in error_msg:
            return "[red]Error:[/red] Model version is outdated. Please update the model version."
        if "model_not_found" in error_msg:
            return "[red]Error:[/red] Invalid model. Available models: " + ", ".join(MODELS.keys())
        return f"[red]Error:[/red] {error_msg}"

def parse_arguments():
    parser = argparse.ArgumentParser(description="CLI GPT - A command-line interface for chat models")
    parser.add_argument("--model", "-m", choices=MODELS.keys(), default="gemini",
                      help="Model to use for chat (default: gemini)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set up command history
    session = PromptSession(history=FileHistory(str(HISTORY_DIR / "command_history")))
    
    # Initialize conversation
    conversation = Conversation(args.model)
    
    console.print(Panel.fit(
        "Welcome to CLI GPT! Type 'exit' to quit.\n"
        "Commands:\n"
        "  /save    - Save the current conversation\n"
        "  /clear   - Clear the conversation history\n"
        "  /model   - Show current model\n"
        "  /switch  - Switch to a different model\n"
        "  /help    - Show this help message",
        border_style="blue",
        padding=(1, 2),
        title="🤖 CLI GPT",
        subtitle=f"Using {args.model} model"
    ))

    try:
        while True:
            query = session.prompt("\n[bold green]You[/bold green] > ")
            
            # Handle commands
            if query.lower() in ["exit", "quit"]:
                conversation.save()
                console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break
            elif query.lower() == "/save":
                conversation.save()
                console.print("[green]Conversation saved![/green]")
                continue
            elif query.lower() == "/clear":
                conversation = Conversation(args.model)
                console.print("[yellow]Conversation history cleared![/yellow]")
                continue
            elif query.lower() == "/model":
                console.print(f"[blue]Current model:[/blue] {args.model} ({MODELS[args.model]['provider']})")
                continue
            elif query.lower() == "/switch":
                available_models = list(MODELS.keys())
                current_index = available_models.index(args.model)
                next_index = (current_index + 1) % len(available_models)
                args.model = available_models[next_index]
                conversation = Conversation(args.model)
                console.print(f"[blue]Switched to model:[/blue] {args.model} ({MODELS[args.model]['provider']})")
                continue
            elif query.lower() == "/help":
                console.print(Panel.fit(
                    "Commands:\n"
                    "  /save    - Save the current conversation\n"
                    "  /clear   - Clear the conversation history\n"
                    "  /model   - Show current model\n"
                    "  /switch  - Switch to a different model\n"
                    "  /help    - Show this help message",
                    title="Help"
                ))
                continue
            
            reply = ask_gpt(query, conversation)
            
            if reply.startswith("[red]Error:"):
                console.print(reply)
            else:
                console.print(format_response(reply))
    
    except KeyboardInterrupt:
        conversation.save()
        console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
    except Exception as e:
        console.print(f"[red]An error occurred:[/red] {str(e)}")
        conversation.save()
