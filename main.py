import requests
from textual.containers import Vertical
import ENV
from rich import print
from textual.app import App 
from textual.widgets import Header, Footer, Static, TextArea
from pathlib import Path



def llm_response(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {ENV.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=data)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



class MainTUI(App):
    with open(f"{Path(__file__).resolve().parent}/CSS.css") as f:
        CSS = f.read()
    def compose(self):
        yield Vertical(
            Header(),
            TextArea("TextArea", id="input", show_cursor=True),
            Static("Static", id="text", expand=True),
            Footer()
        )
    def on_mount(self):
        self.query_one("#input").focus()




def main():
    print("[red]TUI building journey beginns here! [/red]")


if __name__ == "__main__":
    main()
    MainTUI().run()
