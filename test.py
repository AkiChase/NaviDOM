import json
from pathlib import Path

from agent.agent import Agent


def main():
    out_dir = Path("output/test")
    with open(out_dir / "result.json", "r", encoding="utf-8") as f:
        out = json.load(f)
    Agent.save_report(**out, out_dir=out_dir)


if __name__ == "__main__":
    main()
