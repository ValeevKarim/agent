from typing import List, Dict


class ConversationMemory:
    def __init__(self, max_turns = 20):
        self.max_turns = max_turns
        self.conversation = []
        self.context_summary = ""

    def add_message(self, role: str, content: str, name: str = None):
        message = {"role": role, "content": content}
        if name:
            message["name"] = name

        self.conversation.append(message)

        if len(self.conversation) > self.max_turns * 2:  # cutting history len
            if self.conversation[0]["role"] == "system":
                self.conversation = [self.conversation[0]] + self.conversation[-self.max_turns * 2:]
            else:
                self.conversation = self.conversation[-self.max_turns * 2:]

    def get_context(self, max_tokens: int = 4000) -> List[Dict]:  # conversation context
        total_chars = sum(len(str(msg)) for msg in self.conversation)

        if total_chars > max_tokens * 4:  # Rough estimate: 4 chars per token
            recent = self.conversation[-6:]  # Last 3 exchanges
            if self.context_summary:
                summary_msg = {"role": "system", "content": f"Previous context summary: {self.context_summary}"}
                return [summary_msg] + recent
            return recent

        return self.conversation

    def summarize_conversation(self):
        if len(self.conversation) < 4:
            return

        user_queries = []
        assistant_actions = []

        for msg in self.conversation:
            if msg["role"] == "user":
                user_queries.append(msg["content"][:100])
            elif msg["role"] == "assistant" and "tool" in msg.get("name", ""):
                assistant_actions.append(f"{msg.get('name')}: {msg['content'][:50]}")

        summary = f"User asked about: {', '.join(user_queries[-3:])}. "
        if assistant_actions:
            summary += f"Assistant used tools: {', '.join(assistant_actions[-3:])}."

        self.context_summary = summary
        return summary