# utils.py
import re
import json
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat


async def safe_initiate_chat(agent: AssistantAgent, message: str, user_proxy: UserProxyAgent, max_turns: int = 3):
    """Safely initiate chat with error handling using v6.0 API."""
    try:
        # Create a simple two-agent team
        team = RoundRobinGroupChat([user_proxy, agent])
        
        # Run the conversation
        chat_result = await team.run(
            task=message,
            max_turns=max_turns
        )
        
        # Extract the last message content
        if chat_result.messages:
            return chat_result.messages[-1].content
        else:
            return "No response received"
            
    except Exception as e:
        print(f"Error in chat with {agent.name}: {e}")
        return f"Error: {e}"


def safe_initiate_chat_sync(agent: AssistantAgent, message: str, user_proxy: UserProxyAgent, max_turns: int = 3):
    """Synchronous wrapper for safe_initiate_chat."""
    try:
        return asyncio.run(safe_initiate_chat(agent, message, user_proxy, max_turns))
    except Exception as e:
        print(f"Error in sync chat wrapper: {e}")
        return f"Error: {e}"


async def get_agent_response(agent: AssistantAgent, message: str):
    """Get a direct response from an agent."""
    try:
        # Create a text message
        text_message = TextMessage(content=message, source="user")
        
        # Get response from agent
        response = await agent.on_messages([text_message], cancellation_token=None)
        
        if response and hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return "No valid response received"
            
    except Exception as e:
        print(f"Error getting response from {agent.name}: {e}")
        return f"Error: {e}"


def get_agent_response_sync(agent: AssistantAgent, message: str):
    """Synchronous wrapper for get_agent_response."""
    try:
        return asyncio.run(get_agent_response(agent, message))
    except Exception as e:
        print(f"Error in sync response wrapper: {e}")
        return f"Error: {e}"


def load_prompt(name: str) -> str:
    """Loads a prompt from the 'prompts' directory."""
    try:
        with open(f"prompts/{name}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback prompts if files are missing
        fallbacks = {
            "codegen": "You are an expert Python programmer. Write clean, efficient code.",
            "testcase": "You are a QA engineer. Generate comprehensive JSON test cases.",
            "verifier": "You are a code verifier. Analyze test results and find bugs.",
            "corrector": "You are a code corrector. Fix the given code based on test failures.",
            "task_analyzer": "You analyze tasks and recommend reasoning strategies.",
            "symbolic_reasoner": "You translate logic into symbolic representations and abstract reasoning."
        }
        return fallbacks.get(name, f"You are a helpful {name} assistant.")


class CodeExtractor:
        """Enhanced code extraction from LLM responses."""
        
        @staticmethod
        def extract_python_code(text):
            code_block_pattern = r'```python\n(.*?)\n```'
            match = re.search(code_block_pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text  # Fallback to returning the whole text

        @staticmethod
        def extract_pseudocode(text):
            pseudocode_patterns = [
                r'```pseudocode\n(.*?)\n```',
                r'```(.*?)\n```'  # Generic block
            ]
            for pattern in pseudocode_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return text


def extract_json_from_response(response: str) -> str:
    """Extracts a JSON object or array from an agent's string response."""
    match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
    if match:
        return match.group(0)
    return response


def print_test_results(results):
    """Print detailed test results."""
    print("\n📊 DETAILED TEST RESULTS:")
    print("=" * 50)
    if not isinstance(results, list) or not results:
        print("No valid results to display.")
        return False
    
    passed_count = sum(1 for r in results if isinstance(r, dict) and r.get("passed"))
    total_count = len(results)
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            print(f"Test {i+1}: {status}")
            print(f"   - Input: {result.get('input', 'N/A')}")
            print(f"   - Expected: {result.get('expected', 'N/A')}")
            print(f"   - Actual: {result.get('actual', 'N/A')}")
            if result.get("error"):
                print(f"   - Error: {result['error']}")
        else:
            print(f"Test {i+1}: ❌ FAILED - Invalid result format: {result}")
    
    success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n📈 SUMMARY: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")
    return passed_count == total_count


# Utility function to handle both sync and async contexts
def run_agent_task(agent: AssistantAgent, message: str, user_proxy: UserProxyAgent = None, max_turns: int = 3):
    """
    Convenience function that handles both direct agent responses and chat-based interactions.
    """
    if user_proxy:
        return safe_initiate_chat_sync(agent, message, user_proxy, max_turns)
    else:
        return get_agent_response_sync(agent, message)