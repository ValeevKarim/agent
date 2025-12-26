import json
import re
from pathlib import Path
import datetime
from ollama import chat
from tools.schemas import get_tools_for_ollama
from tools.implementations import ToolExecutor
from prompts import build_initial_prompt, build_reasoning_prompt


class ToolUsingAgent:
    def __init__(self, config_path="config.json", use_reasoning=True):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.tool_executor = ToolExecutor(config_path)

        self.tools = get_tools_for_ollama()  # getting tools for ollama
        tool_names = [t["function"]["name"] for t in self.tools]

        self.use_reasoning = use_reasoning

        self.system_prompt = build_initial_prompt(tool_names)

        self.history = [  # make conversation using system prompt
            {"role": "system", "content": self.system_prompt}
        ]

        print("Agent Initialized")
        print(f"Tools: {', '.join(tool_names)}")
        print(f"Reasoning: {'Enabled' if use_reasoning else 'Disabled'}")
        print("Type 'exit' to quit, 'help' for commands\n")
        print("Langtrace monitoring enabled")

    def _extract_tool_calls(self, text: str):
        if not text:
            return None

        json_patterns = [  # find json files in text
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',  # Generic code block
            r'(\{.*?"tool_calls".*?\})',  # JSON with tool_calls
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match.strip())
                        if "tool_calls" in data:
                            return data["tool_calls"]
                    except json.JSONDecodeError:
                        continue

        try:
            data = json.loads(text.strip())
            if "tool_calls" in data:
                return data["tool_calls"]
        except json.JSONDecodeError:
            pass

        if '"tool_calls"' in text:  # if nothing  found try to find anything that loo0ks like tool calls
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                try:
                    data = json.loads(json_str)
                    if "tool_calls" in data:
                        return data["tool_calls"]
                except:
                    pass

        return None

    def _reason_about_query(self, query: str) -> str:
        if not self.use_reasoning:
            return None

        print("Thinking about the best approach...")

        reasoning_prompt = build_reasoning_prompt(query)

        reasoning_response = chat(
            model=self.config.get("model_name", "gemma3:1b"),
            messages=[
                {"role": "system", "content": "You are a strategic thinker. Analyze requests and plan tool usage."},
                {"role": "user", "content": reasoning_prompt}
            ]
        )

        reasoning = reasoning_response["message"]["content"]
        print(f"Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"Reasoning: {reasoning}")

        return reasoning

    def process_message(self, user_input: str) -> str:
            reasoning = self._reason_about_query(user_input)

            self.history.append({"role": "user", "content": user_input})  # user message to hisstory

            messages = self.history.copy()  # add reasoning if exists
            if reasoning:
                messages.insert(-1, {"role": "assistant", "content": f"Thought: {reasoning}"})

            print("Generating response...")

            try:
                response = chat(
                    model=self.config.get("model_name", "gemma3:1b"),
                    messages=messages,
                    stream=False
                )

                message = response["message"]
                response_text = message.content

                print(f"Raw response: {response_text[:200]}...")

                tool_calls = self._extract_tool_calls(response_text)  # check if tools calls exist

                if tool_calls:
                    print(f"Detected {len(tool_calls)} tool call(s)")
                    return self._execute_tool_calls(tool_calls, response_text)
                else:
                    # check if need to use tools to respond
                    if any(phrase in response_text.lower() for phrase in
                           ['here is the json', 'json response', 'tool_calls', 'i will use']):
                        print("Warning: LLM is describing tools instead of using them")
                        tool_calls = self._extract_tool_calls(response_text)
                        if tool_calls:
                            return self._execute_tool_calls(tool_calls, response_text)
                        else:
                            return "I need to use tools to help you. Let me try again with clearer instructions."

                    cleaned_response = self._clean_response(response_text)
                    self.history.append({"role": "assistant", "content": cleaned_response})
                    return cleaned_response

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"Error: {error_msg}")
                return error_msg

    def _clean_response(self, text: str) -> str:  # clean response from json if left any
        lines = text.split('\n')
        clean_lines = []

        in_json = False
        for line in lines:
            if line.strip().startswith('{') and '"tool_calls"' in line:
                in_json = True
            elif in_json and line.strip().endswith('}'):
                in_json = False
                continue

            if not in_json:
                clean_lines.append(line)

        cleaned = '\n'.join(clean_lines).strip()

        if not cleaned:
            return "I've analyzed your request. Would you like me to search the codebase or examine specific files to help you better?"

        return cleaned

    def _execute_tool_calls(self, tool_calls, original_response: str) -> str:

        print("\n" + "="*50)
        print("EXECUTING TOOLS")
        print("="*50)

        all_tool_results = []

        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            print(f"\n[{i}] Tool: {tool_name}")
            print(f"    Arguments: {json.dumps(arguments, indent=2)}")

            tool_result = self.tool_executor.execute_tool(tool_name, arguments)

            all_tool_results.append({
                "tool": tool_name,
                "result": tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
            })

            print(f"    Result: {tool_result[:200]}..." if len(tool_result) > 200 else f"    Result: {tool_result}")

            self.history.append({"role": "tool",  # add tools to history
                "content": tool_result,
                "name": tool_name
            })

        print("="*50)
        print("PROCESSING TOOL RESULTS")
        print("="*50)

        # Get final response with tool results
        final_response = chat(
            model=self.config.get("model_name", "gemma3:1b"),
            messages=self.history,
            tools=self.tools
        )

        final_message = final_response["message"]
        final_content = self._clean_response(final_message.content)

        self.history.append({"role": "assistant", "content": final_content})  # expand history with response

        response_parts = []  # build a response

        if original_response and not original_response.strip().startswith('{'):
            cleaned_original = self._clean_response(original_response)
            if cleaned_original:
                response_parts.append(cleaned_original)

        if all_tool_results:
            response_parts.append("\n" + "="*50)
            response_parts.append("TOOLS USED:")
            for tr in all_tool_results:
                response_parts.append(f"\n* {tr['tool']}: {tr['result']}")

        response_parts.append("\n" + "="*50)
        response_parts.append(f"\n{final_content}")

        return "\n".join(response_parts)

    def chat_loop(self):
        print("="*60)
        print("\nHow can I help u:")
        print("="*60)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! Happy coding!")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'clear':
                    self.history = [{"role": "system", "content": self.system_prompt}]
                    print("\nConversation cleared")
                    continue

                if not user_input:
                    continue

                response = self.process_message(user_input)  # processing user message

                print(f"\n{'='*60}")
                print(f"CodeCraft:")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\nError: {str(e)}")

    def _show_help(self):
        help_text = "no help for you"

        print(help_text)


def main():
    db_dir = Path(__file__).parent / "faiss_db"
    if not (db_dir / "repo.index").exists():
        print("Warning: No FAISS index found.")
        print("Please run: python index_repo.py")
        return

    print("Initializing Agent...")
    agent = ToolUsingAgent(use_reasoning=True)
    agent.chat_loop()


if __name__ == "__main__":
    main()
