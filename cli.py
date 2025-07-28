from dotenv import load_dotenv
import requests
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
import typer
load_dotenv()
# --- Configuration ---
BASE_URL = os.getenv("HACKRX_API_URL", "http://localhost:8000")
API_ENDPOINT = "/hackrx/run"
# This is the hardcoded token from the problem statement
AUTH_TOKEN = "e66c2e8eb6884ded2c7177421784e760b34b9297bfebc20a2a272cc63357270d"
DEFAULT_REQUEST_FILE = "request.json"

# --- Rich Console for good looking output ---
console = Console()

cli_app = typer.Typer(help="A simple CLI to run the HackRx 6.0 submission.")

@cli_app.command()
def run(
    request_file: str = typer.Argument(
        DEFAULT_REQUEST_FILE,
        help="Path to the JSON file containing the request body."
    )
):
    """
    Sends the request from a JSON file to the HackRx API endpoint.
    """
    console.print(Panel(f"[bold]🚀 Starting HackRx Submission Test[/bold]", border_style="green"))

    # --- 1. Load and Validate Request File ---
    try:
        with open(request_file, 'r') as f:
            payload = json.load(f)
        
        # Basic validation
        if 'documents' not in payload or 'questions' not in payload:
            console.print(f"[bold red]❌ Error: The file '{request_file}' is missing required keys 'documents' or 'questions'.[/bold red]")
            raise typer.Exit()
            
        console.print(f"✅ Loaded request from: [cyan]{request_file}[/cyan]")
        console.print(f"📄 Document URL: [link={payload['documents']}]Click to view[/link]")
        console.print(f"❓ Number of questions: [bold yellow]{len(payload['questions'])}[/bold yellow]")

    except FileNotFoundError:
        console.print(f"[bold red]❌ Error: Request file not found at '{request_file}'.[/bold red]")
        raise typer.Exit()
    except json.JSONDecodeError:
        console.print(f"[bold red]❌ Error: Could not parse '{request_file}' as valid JSON.[/bold red]")
        raise typer.Exit()

    # --- 2. Prepare and Send API Request ---
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    url = f"{BASE_URL}{API_ENDPOINT}"
    console.print(f"📡 Sending request to: [bold blue]{url}[/bold blue]")

    spinner = Spinner("dots", text=" [bold green]Processing document and generating answers... This may take a moment.[/bold green]")
    try:
        with Live(spinner, console=console, transient=True, refresh_per_second=20) as live:
            response = requests.post(url, headers=headers, json=payload, timeout=300) # 5 minute timeout
        
        # --- 3. Process and Display Response ---
        if response.status_code == 200:
            console.print("[bold green]✅ Request successful! Received answers.[/bold green]")
            try:
                response_data = response.json()
                answers = response_data.get("answers", [])
                
                for i, (question, answer) in enumerate(zip(payload["questions"], answers)):
                    console.print(Panel(f"[bold]Question {i+1}:[/bold] [dim]{question}[/dim]\n\n[bold]Answer:[/bold] {answer}", 
                                  title=f"Result {i+1}/{len(answers)}",
                                  border_style="cyan",
                                  expand=True))

            except json.JSONDecodeError:
                console.print("[bold red]❌ Error: Could not parse the server's response as JSON.[/bold red]")
                console.print("Raw Response:", response.text)
        else:
            console.print(f"[bold red]❌ API Error: {response.status_code}[/bold red]")
            try:
                console.print(Panel(json.dumps(response.json(), indent=2), title="Error Details", border_style="red"))
            except json.JSONDecodeError:
                 console.print("Raw Error Response:", response.text)

    except requests.exceptions.RequestException as e:
        console.print(f"\n[bold red]❌ Connection Error:[/bold red] Could not connect to the API at {BASE_URL}.")
        console.print(f"Details: {e}")

if __name__ == "__main__":
    cli_app()