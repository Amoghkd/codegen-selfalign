# tools.py
import json
from duckduckgo_search import DDGS

class PythonCodeRunner:
    """Real Python code runner that executes generated code."""
    def __init__(self):
        self.namespace = {}
    
    def run_code_with_tests(self, python_code, test_cases):
        results = []
        try:
            if isinstance(test_cases, str):
                test_data = json.loads(test_cases)
            else:
                test_data = test_cases
            if not isinstance(test_data, list):
                test_data = [test_data]
            exec_namespace = {}
            exec(python_code, exec_namespace)
            main_function = None
            for name, obj in exec_namespace.items():
                if callable(obj) and not name.startswith('_'):
                    main_function = obj
                    break
            if not main_function:
                return [{"error": "No callable function found in generated code", "passed": False}]
            for i, test_case in enumerate(test_data):
                try:
                    if isinstance(test_case, dict):
                        inputs = test_case.get("input", test_case.get("inputs", []))
                        expected = test_case.get("expected", test_case.get("output"))
                    else:
                        inputs = []
                        expected = test_case
                    if isinstance(inputs, list):
                        actual_output = main_function(*inputs)
                    elif isinstance(inputs, dict):
                        actual_output = main_function(**inputs)
                    else:
                        actual_output = main_function(inputs)
                    passed = actual_output == expected
                    results.append({"test_id": i, "passed": passed, "input": inputs, "expected": expected, "actual": actual_output, "error": None})
                except Exception as e:
                    results.append({"test_id": i, "passed": False, "input": locals().get('inputs', "Unknown"), "expected": locals().get('expected', "Unknown"), "actual": None, "error": str(e)})
            return results
        except Exception as e:
            return [{"error": f"Code execution failed: {str(e)}", "passed": False}]

    def run_code_safely(self, python_code: str) -> dict:
        """Executes Python code safely without requiring test cases."""
        try:
            exec_namespace = {}
            exec(python_code, exec_namespace)
            return {"success": True, "message": "Code executed successfully.", "namespace": exec_namespace}
        except Exception as e:
            return {"success": False, "error": str(e)}
     
def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the top results."""
    print(f"🔎 Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            formatted_results = "\n\n".join([f"Title: {res['title']}\nSnippet: {res['body']}\nURL: {res['href']}" for res in results])
            return f"Search results for '{query}':\n\n{formatted_results}"
    except Exception as e:
        return f"An error occurred during web search: {e}"