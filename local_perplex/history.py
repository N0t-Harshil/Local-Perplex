import json
import os
from pathlib import Path
from datetime import datetime

class HistoryManager:
    def __init__(self, history_dir: str = None):
        if history_dir is None:
            history_dir = os.getenv("LOCAL_PERPLEX_HISTORY_DIR")
            if history_dir is None:
                project_root = Path(__file__).parent.parent
                history_dir = project_root / "history"
        else:
            history_dir = Path(history_dir)
        
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, question: str, answer: str, sources: list, tag: str = "General",
                     metrics: dict | None = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_q = "".join([c if c.isalnum() else "_" for c in question[:30]])
        filename = f"{timestamp}_{safe_q}_{os.urandom(2).hex()}.json"

        data = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "sources": sources,
            "tag": tag,
            "metrics": metrics or {},
        }

        with open(self.history_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_all_sessions(self):
        sessions = []
        for file in sorted(self.history_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        return sessions
