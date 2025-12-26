TOOL_SCHEMAS = [
    {
        "name": "search_codebase",
        "description": "Search through the code repository using semantic search. Use this when you need to find relevant code snippets or understand the codebase structure.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "What to search for (e.g., 'function that handles authentication', 'database connection code')"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this when you need to see exactly what's in a specific file.",
        "parameters": {
            "file_path": {
                "type": "string",
                "description": "Path to the file (e.g., 'src/utils.py', 'config.json')"
            }
        }
    },
    {
        "name": "modify_file",
        "description": "Make changes to a code file. Use this when asked to fix, update, or improve code.",
        "parameters": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to modify"
            },
            "change_description": {
                "type": "string",
                "description": "Description of what needs to be changed (be specific)"
            },
            "old_code": {
                "type": "string",
                "description": "The exact code to replace (if replacing). Leave empty for new code."
            },
            "new_code": {
                "type": "string",
                "description": "The new code to write"
            },
            "line_number": {
                "type": "integer",
                "description": "Line number where to make the change (if known)",
                "optional": True
            }
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory. Use this to explore the project structure.",
        "parameters": {
            "directory_path": {
                "type": "string",
                "description": "Directory to list (default: project root)",
                "default": "."
            }
        }
    }
]


def get_tools_for_ollama():
    ollama_tools = []

    for schema in TOOL_SCHEMAS:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        param_name: {
                            "type": param_info.get("type", "string"),
                            "description": param_info.get("description", ""),
                            **({"default": param_info["default"]} if "default" in param_info else {})
                        }
                        for param_name, param_info in schema["parameters"].items()
                    },
                    "required": [
                        name for name, info in schema["parameters"].items()
                        if not info.get("optional", False) and "default" not in info
                    ]
                }
            }
        })

    return ollama_tools