from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
# from autogen_agentchat.base import ConversableAgent
from config import make_llm_config, FREE_MODELS
from utils import load_prompt
from tools import PythonCodeRunner,web_search
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
def create_all_agents() -> Dict[str, Any]:
    """Creates and returns all agents with assigned models and prompts."""
    
    llm_configs = {
        "coding": make_llm_config(FREE_MODELS["coding"]),
        "reasoning": make_llm_config(FREE_MODELS["reasoning"]),
        "general": make_llm_config(FREE_MODELS["general"]),
    }
    code_runner = PythonCodeRunner()
    agents = {
        "task_analyzer": AssistantAgent(
            name="task_analyzer",
            model_client=llm_configs["reasoning"],
            system_message=load_prompt("task_analyzer") or "You analyze tasks and recommend reasoning strategies."
        ),

        "codegen": AssistantAgent(
            name="codegen",
            model_client=llm_configs["coding"],
            system_message=load_prompt("codegen") or "Write Python code based on tasks or plans" ,
            # tools=[code_runner.run_code_safely, PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))] ,
            # reflect_on_tool_use=True
        ),

        "critiquer": AssistantAgent(
            name="critiquer",                           
            model_client=llm_configs["reasoning"],
            system_message=load_prompt("critiquer") or "You provide critical feedback on code and logic plans."
        ),

        "testwriter": AssistantAgent(
            name="testwriter",
            model_client=llm_configs["coding"],
            system_message=load_prompt("testcase") or "Write comprehensive test cases for Python functions."
        ),

        "corrector": AssistantAgent(
            name="corrector",
            model_client=llm_configs["coding"],
            system_message=load_prompt("corrector") or "Improve or fix Python code."
        ),

        "reasoner": AssistantAgent(
            name="reasoner",
            model_client=llm_configs["reasoning"],
            system_message="You provide detailed logical analysis and improvements."
        ),

        "quick_reasoner": AssistantAgent(
            name="quick_reasoner",
            model_client=llm_configs["reasoning"],
            system_message="You give fast, simple feedback on logic plans."
            
        ),

        "logical_reasoner": AssistantAgent(
            name="logical_reasoner",
            model_client=llm_configs["reasoning"],
            system_message="You deeply analyze problems and decompose them logically."
        ),

        "symbolic_reasoner": AssistantAgent(
            name="symbolic_reasoner",
            model_client=llm_configs["reasoning"],
            system_message=load_prompt("symbolic_reasoner") or "You translate logic into symbolic representations and abstract reasoning."
        ),

        "user_proxy": UserProxyAgent(
            name="user_proxy",
        )
    }

    return agents