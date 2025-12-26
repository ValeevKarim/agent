import json
import shutil
from pathlib import Path
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer


from agent import retrieve_chunks

from memory import ConversationMemory


class ToolExecutor:
    def __init__(self, config_path="config.json", memory: ConversationMemory = None):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        MODEL_NAME = self.config["model_name"]
        self.top_k = int(self.config["top_k"])
        self.db_dir = Path(__file__).parent.parent / "faiss_db"
        self.index_path = self.db_dir / "repo.index"
        self.docs_path = self.db_dir / "docs.json"

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

        if self.docs_path.exists():
            with open(self.docs_path, 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
        else:
            self.docs = []

        self.embed_model = SentenceTransformer(r"C:\all-MiniLM-L6-v2")

        self.change_history = []  # Tracking changes
        self.memory = memory or ConversationMemory()

    def search_codebase(self, query: str, top_k) -> str:
        if not self.index:
            return "ERROR: No index found. Please run index_repo.py first."

        print(f"[Tool] Searching for: '{query}'")
        chunks = retrieve_chunks(query, top_k)

        if not chunks:
            return "No results found."

        if self.memory and self.memory.context_summary:  # using conversation context to improve search
            contextual_query = f"{query}. Previous context: {self.memory.context_summary}"
            print(f"[Tool] Contextual search: {contextual_query[:100]}...")

        results = []
        for i, (chunk, meta) in enumerate(chunks, 1):
            results.append(f"\n--- Result {i} ---")
            results.append(f"File: {meta['path']}")
            if 'score' in meta:
                results.append(f"Relevance: {meta['score']:.3f}")
            results.append(f"\n{chunk[:500]}..." if len(chunk) > 500 else f"\n{chunk}")

        return "\n".join(results)

    def read_file(self, file_path: str) -> str:
        try:
            path = Path(file_path)
            if not path.exists():
                repo_path = Path(self.config["repo_path"])
                path = repo_path / file_path

            if not path.exists():
                return f"ERROR: File not found: {file_path}"

            content = path.read_text(encoding='utf-8')

            file_info = f"File: {path}\nSize: {len(content)} chars, {len(content.splitlines())} lines\n"
            file_info += "=" * 50 + "\n"

            return file_info + content

        except Exception as e:
            return f"ERROR reading file: {str(e)}"

    def modify_file(self, file_path: str, change_description: str,
                    new_code: str, old_code: str = "", line_number: int = None) -> str:

        if not self.config.get("allow_modifications", False):
            return "ERROR: File modifications are disabled in config. Set 'allow_modifications' to true."

        try:
            path = Path(file_path)
            if not path.exists():
                repo_path = Path(self.config["repo_path"])
                path = repo_path / file_path

            if not path.exists():
                return f"ERROR: File not found: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            backup_dir = Path(self.config.get("backup_dir", "./backups"))
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{path.name}.backup_{timestamp}"
            shutil.copy2(path, backup_path)

            if line_number: # find where to make the change
                if line_number > len(lines):
                    lines.append(new_code + "\n")
                    change_type = "append"
                else:
                    lines.insert(line_number - 1, new_code + "\n")
                    change_type = "insert"
            elif old_code:                 # replace specific code does not work
                full_content = ''.join(lines)
                if old_code in full_content:
                    new_content = full_content.replace(old_code, new_code)
                    lines = new_content.splitlines(keepends=True)
                    change_type = "replace"
                else:
                    return f"ERROR: Could not find the specified code to replace:\n{old_code}"
            else:  # add to end
                lines.append("\n" + new_code)
                change_type = "append"

            if self.config.get("require_confirmation", True):
                print(f"\n[CONFIRMATION NEEDED]")
                print(f"File: {path}")
                print(f"Change: {change_description}")
                print(f"Type: {change_type}")
                print(f"\nNew code:\n{new_code}")

                response = input("\nApply this change? (y/n): ").strip().lower()
                if response != 'y':
                    return "Change cancelled by user."

            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)  # do changes

            self.change_history.append({  # addidng changes
                "timestamp": datetime.now().isoformat(),
                "file": str(path),
                "description": change_description,
                "backup": str(backup_path),
                "type": change_type
            })

            return f"SUCCESS: Modified {path}\nBackup saved to: {backup_path}\nChange: {change_description}"

        except Exception as e:
            return f"ERROR modifying file: {str(e)}"

    def list_files(self, directory_path: str = ".") -> str:  # list files and directories
        try:
            base_path = Path(self.config["repo_path"])
            target_path = base_path / directory_path

            if not target_path.exists():
                return f"ERROR: Directory not found: {directory_path}"

            results = [f"Directory: {target_path}"]

            dirs = [d for d in target_path.iterdir() if d.is_dir()]
            if dirs:
                results.append("\n Directories:")
                for d in sorted(dirs)[:20]:  # Limit to 20
                    results.append(f"  {d.name}/")

            files = [f for f in target_path.iterdir() if f.is_file()]
            if files:
                results.append("\n📄 Files:")
                for f in sorted(files)[:30]:  # Limit to 30
                    size = f.stat().st_size
                    results.append(f"  {f.name} ({size} bytes)")

            if len(dirs) > 20 or len(files) > 30:
                results.append(f"\n... and {max(0, len(dirs) - 20) + max(0, len(files) - 30)} more items")

            return "\n".join(results)

        except Exception as e:
            return f"ERROR listing files: {str(e)}"

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "search_codebase":
            return self.search_codebase(**arguments)
        elif tool_name == "read_file":
            return self.read_file(**arguments)
        elif tool_name == "modify_file":
            return self.modify_file(**arguments)
        elif tool_name == "list_files":
            return self.list_files(**arguments)
        else:
            return f"ERROR: Unknown tool '{tool_name}'"