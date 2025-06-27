# main.py
import json
import time
from agents import create_all_agents
from engine import LLMTaskAnalyzer, ReasoningPipelines
from tools import PythonCodeRunner, web_search
from utils import CodeExtractor, extract_json_from_response, print_test_results, safe_initiate_chat_sync, run_agent_task

def get_user_choice(recommended_strategy: str) -> str:
    print("\n🤔 CHOOSE A REASONING STRATEGY")
    print("-" * 40)
    print(f"🤖 LLM Recommends: [{recommended_strategy}]")
    print("1: Code-First (Simple, direct tasks)")
    print("2: Pseudocode-First (Medium, algorithmic tasks)")
    print("3: Neuro-Symbolic (Complex, critical tasks)")
    
    strategy_map = {
        "1": "CODE_FIRST",
        "2": "PSEUDOCODE_FIRST",
        "3": "NEURO_SYMBOLIC"
    }
    
    while True:
        choice = input("Enter your choice [1, 2, 3] or press Enter to accept recommendation: ").strip()
        if not choice:
            return recommended_strategy
        if choice in strategy_map:
            return strategy_map[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")

def generate_and_run_tests(task: str, code: str, agents: dict) -> tuple[bool, str]:
    print("\n🧪 GENERATING AND RUNNING TESTS...")
    user_proxy = agents["user_proxy"]

    tc_prompt = f"""Generate comprehensive test cases for this task.
Return a JSON array of test cases with 'input' and 'expected' fields.
Include edge cases and typical scenarios.

Task: {task}
"""
    tc_response = safe_initiate_chat_sync(agents["testwriter"], tc_prompt, user_proxy)
    test_cases_str = extract_json_from_response(tc_response)

    if not test_cases_str:
        print("⚠️ Could not generate test cases, skipping verification")
        return True, code
    
    print(f"📋 Test cases generated.")
    
    code_runner = PythonCodeRunner()
    code_extractor = CodeExtractor()
    python_code = code_extractor.extract_python_code(code)

    if not python_code.strip():
        print("❌ No valid Python code found")
        return False, code

    results = code_runner.run_code_with_tests(python_code, test_cases_str)
    success = print_test_results(results)
    return success, python_code

def final_correction_loop(task: str, initial_code: str, test_results: str, agents: dict) -> str:
    print("\n🔧 FINAL CORRECTION LOOP...")
    current_code = initial_code
    user_proxy = agents["user_proxy"]

    for attempt in range(3):
        print(f"🔄 Correction attempt {attempt + 1}/3")

        correction_prompt = f"""Fix this code based on test failures.
Return only the corrected Python code wrapped in ```python ... ```.

Task: {task}
Current Code: {current_code}
Test Results: {test_results}
"""
        corrected_solution = safe_initiate_chat_sync(agents["corrector"], correction_prompt, user_proxy)
        code_extractor = CodeExtractor()
        corrected_code = code_extractor.extract_python_code(corrected_solution)

        if not corrected_code:
            print("⚠️ Could not extract corrected code")
            continue

        code_runner = PythonCodeRunner()
        test_cases_str = extract_json_from_response(test_results)
        if test_cases_str:
            results = code_runner.run_code_with_tests(corrected_code, test_cases_str)
            if print_test_results(results):
                print("✅ Correction successful!")
                return corrected_code

        current_code = corrected_code

    print("⚠️ Could not fully correct the code after 3 attempts")
    return current_code

def main():
    print("🚀 LLM-DRIVEN MULTI-STRATEGY CODE GENERATION SYSTEM 🚀")
    print("=" * 70)

    task = input("📝 Enter your programming task (or press Enter for default): ").strip()
    if not task:
        task = "Create a Python function that finds the longest palindromic substring in a given string."

    print(f"\n🎯 Task: {task}")
    print("=" * 70)

    print("🔧 Setting up agents and tools...")
    agents = create_all_agents()
    user_proxy = agents["user_proxy"]
    
    # Note: Function registration might work differently in v6.0
    # This may need adjustment based on your specific v6.0 setup
    try:
        user_proxy.register_function(function_map={"web_search": web_search})
    except AttributeError:
        print("⚠️ Function registration method may need adjustment for v6.0")

    print("\n1️⃣ ANALYZING TASK COMPLEXITY...")
    analyzer = LLMTaskAnalyzer(agents)
    analysis = analyzer.analyze_task(task)

    print("🧠 Analysis Results:")
    print(f"   - Strategy: {analysis.get('reasoning_strategy', 'N/A')}")
    print(f"   - Complexity: {analysis.get('complexity', 'N/A')}/10")
    print(f"   - Reasoning: {analysis.get('explanation', 'N/A')}")

    recommended_strategy = analysis.get("reasoning_strategy", "CODE_FIRST").upper()
    chosen_strategy = get_user_choice(recommended_strategy)
    print(f"\n✅ Selected Strategy: [{chosen_strategy}]")

    print("\n2️⃣ EXECUTING REASONING PIPELINE...")
    pipelines = ReasoningPipelines(agents)

    pipeline_map = {
        "CODE_FIRST": pipelines.code_first_pipeline,
        "PSEUDOCODE_FIRST": pipelines.pseudocode_first_pipeline,
        "NEURO_SYMBOLIC": pipelines.neuro_symbolic_pipeline,
    }

    selected_pipeline = pipeline_map[chosen_strategy]
    solution = selected_pipeline(task)

    if not solution or solution.strip() == "":
        print("Pipeline failed to generate a solution")
        return

    print("\n3️⃣ TESTING AND VERIFICATION...")
    test_success, final_code = generate_and_run_tests(task, solution, agents)

    if not test_success:
        print("\n⚠️ Tests failed, attempting final corrections...")
        code_runner = PythonCodeRunner()
        code_extractor = CodeExtractor()
        tc_response = safe_initiate_chat_sync(agents["testwriter"], f"Generate test cases for: {task}", user_proxy)
        test_cases_str = extract_json_from_response(tc_response)

        if test_cases_str:
            results = code_runner.run_code_with_tests(final_code, test_cases_str)
            final_code = final_correction_loop(task, final_code, str(results), agents)

    print("\n" + "=" * 70)
    print("🎉 FINAL RESULTS")
    print("=" * 70)
    print(f"Strategy Used: [{chosen_strategy}]")
    print(f"Task: {task}")
    print("\n✅ Final Solution:")
    print("-" * 40)
    if final_code and final_code.strip():
        print(final_code)
    else:
        print("❌ No valid solution was generated")
    print("-" * 40)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()