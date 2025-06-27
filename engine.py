# engine.py
import json
import re
import asyncio
import re
import json
from utils import CodeExtractor, extract_json_from_response, run_agent_task,safe_initiate_chat
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.messages import TextMessage




class ReasoningPipelines:
    def __init__(self, agents):
        self.agents = agents
        self.user_proxy = agents["user_proxy"]

    

    def code_first_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Code-First] Pipeline with GraphFlow...")
        
        def build_flow():
            builder = DiGraphBuilder()
            # builder.add_node(self.user_proxy)
            builder.add_node(self.agents["codegen"])
            builder.add_node(self.agents["critiquer"])
            builder.add_node(self.agents["corrector"])
            # builder.add_edge(self.user_proxy, self.agents["codegen"])
            builder.add_edge(self.agents["codegen"], self.agents["critiquer"])
            builder.add_edge(self.agents["critiquer"], self.agents["corrector"])
            return GraphFlow(builder.get_participants(), graph=builder.build())
        
        system_prompt = f"""You are solving the following Python task. Try to generate correct code.
        Task: {task}
        Return code only in a ```python ... ``` block."""
        
        current_solution = None
        
        async def run_flow_capture_code():
            final_code = ""
            all_messages = []
            flow = build_flow()
            
            try:
                async for event in flow.run_stream(task=system_prompt):
                    print(f"🔄 Event type: {type(event).__name__}")
                    
                    if isinstance(event, TextMessage):
                        content = event.content
                        all_messages.append(content)
                        print(f"📝 Message: {content[:100]}...")
                        
                        if "```python" in content or "```" in content:
                            extracted = CodeExtractor.extract_python_code(content)
                            if extracted and len(extracted) > 10:  # Basic validation
                                final_code = extracted
                                print(f"✅ Code extracted: {len(extracted)} characters")
                    
                    # Handle other event types that might contain the final result
                    elif hasattr(event, 'content') and event.content:
                        content = str(event.content)
                        all_messages.append(content)
                        if "```python" in content or "```" in content:
                            extracted = CodeExtractor.extract_python_code(content)
                            if extracted and len(extracted) > 10:
                                final_code = extracted
            
            except Exception as e:
                print(f"Flow execution error: {e}")
                # Try to extract code from any collected messages
                for msg in all_messages:
                    if "```" in msg:
                        extracted = CodeExtractor.extract_python_code(msg)
                        if extracted and len(extracted) > 10:
                            final_code = extracted
                            break
            
            # If no code found, try the last substantial message
            if not final_code and all_messages:
                # Use the last message that might contain code
                for msg in reversed(all_messages):
                    if len(msg.strip()) > 10:
                        final_code = CodeExtractor.extract_python_code(msg)
                        if final_code != msg:  # Code was extracted
                            break
                else:
                    final_code = all_messages[-1] if all_messages else ""
            
            return final_code
        
        for attempt in range(3):
            print(f"\n🔁 Attempt {attempt + 1}")
            
            try:
                # Add timeout to prevent hanging
                current_solution = asyncio.run(
                    asyncio.wait_for(run_flow_capture_code(), timeout=120.0)
                )
                
            except asyncio.TimeoutError:
                print(f"⏰ Timeout in attempt {attempt + 1}")
                continue
            except Exception as e:
                print(f"Error in attempt {attempt + 1}: {e}")
                if "RateLimitError" in str(e) or "429" in str(e):
                    print("❌ Rate limit exceeded. Please add credits or wait for reset.")
                    break
                continue
            
            if not current_solution or len(current_solution.strip()) < 10:
                print("⚠️ No substantial solution generated in this attempt")
                continue
            
            print(f"\n📨 Attempted Code ({len(current_solution)} chars):")
            print(f"{current_solution[:200]}...")
            
            # Extract critique from reasoner's last response using utils
            critique_prompt = f"""Critique this code implementation. Return JSON only:
            {{
            "score": X,
            "issues": [...],
            "fixes": [...]
            }}

            Task: {task}
            Code:
            {current_solution}
            """
            
            try:
                # Use the utils function - this returns a string, not a coroutine
                critique_response = run_agent_task(
                    self.agents["reasoner"], 
                    critique_prompt, 
                    self.user_proxy,
                    max_turns=2
                )
                
                print(f"🧠 Critique response: {critique_response[:200]}...")
                
                # Extract JSON from response using utils
                json_str = extract_json_from_response(critique_response)
                
                try:
                    critique = json.loads(json_str)
                    score = critique.get("score", 0)
                    print(f"\n🧠 Score: {score}/10")
                    
                    if score >= 8:
                        print(f"\n✅ Final score {score}/10 — Code accepted.")
                        return current_solution
                    
                    print("\n🛠️ Retrying refinement...")
                    # Update the system prompt with feedback for next iteration
                    issues = critique.get("issues", [])
                    fixes = critique.get("fixes", [])
                    if issues or fixes:
                        feedback = f"\nPrevious issues: {issues}\nSuggested fixes: {fixes}"
                        system_prompt = f"""{system_prompt}\n{feedback}
                        
                        Please improve the code based on the feedback above."""
                        
                except json.JSONDecodeError as e:
                    print(f"\n⚠️ JSON parsing error: {e}")
                    print(f"Raw response: {json_str}")
                    # Continue to next attempt
                    
            except Exception as e:
                print(f"\n⚠️ Critique generation failed: {e}")
                if "RateLimitError" in str(e) or "429" in str(e):
                    print("❌ Rate limit exceeded during critique.")
                    break
                # Continue to next attempt
        
        print("\n❌ Max retries reached. Returning last version.")
        return current_solution or "# No code returned."
    def pseudocode_first_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Pseudocode-First] Pipeline...")
        plan_prompt = f"""Create pseudocode for solving this task.
Wrap in ```pseudocode ... ```.

Task: {task}
"""
        pseudocode_plan = safe_initiate_chat(self.agents["reasoner"], plan_prompt, self.user_proxy)
        refined_plan = self._collaborative_reasoning(task, pseudocode_plan)
        return self._implement_with_loop(task, refined_plan)

    def neuro_symbolic_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Neuro-Symbolic] Pipeline...")

        decomp_prompt = f"""Decompose this problem logically:
1. Key logic
2. Sub-problems
3. Constraints

Task: {task}
"""
        logic_analysis = safe_initiate_chat(self.agents["logical_reasoner"], decomp_prompt, self.user_proxy)

        symbolic_prompt = f"""Based on this analysis, generate symbolic representation and pseudocode.
Analysis: {logic_analysis}
"""
        symbolic_plan = safe_initiate_chat(self.agents["symbolic_reasoner"], symbolic_prompt, self.user_proxy)
        refined_plan = self._collaborative_reasoning(task, symbolic_plan)
        return self._implement_with_loop(task, refined_plan)

    def _collaborative_reasoning(self, task: str, initial_plan: str) -> str:
        current_plan = initial_plan
        for i in range(2):
            analysis_prompt = f"""Analyze and improve this plan for logic and edge cases.

Task: {task}
Plan: {current_plan}
"""
            detailed = safe_initiate_chat(self.agents["reasoner"], analysis_prompt, self.user_proxy)

            quick = safe_initiate_chat(
                self.agents["quick_reasoner"],
                f"What's the biggest flaw and quick fix in this plan?\n{current_plan}",
                self.user_proxy
            )

            merge_prompt = f"""Merge these analyses into a final refined plan.

Task: {task}
Detailed Analysis: {detailed}
Quick Feedback: {quick}
"""
            current_plan = safe_initiate_chat(self.agents["reasoner"], merge_prompt, self.user_proxy)
        return current_plan

    def _implement_with_loop(self, task: str, plan: str) -> str:
        impl_prompt = f"""Implement this plan in Python.
Plan:
{plan}
Return only the code in ```python ... ``` block.
"""
        current_code = safe_initiate_chat(self.agents["codegen"], impl_prompt, self.user_proxy)

        for attempt in range(3):
            critique_prompt = f"""Critique this code implementation. Return JSON: 
{{"score": X, "issues": [...], "fixes": [...]}}

Task: {task}
Plan: {plan}
Code:
{current_code}
"""
            critique_response = safe_initiate_chat(self.agents["reasoner"], critique_prompt, self.user_proxy)

            try:
                match = re.search(r'\{.*\}', critique_response, re.DOTALL)
                critique = json.loads(match.group()) if match else {}
                score = critique.get("score", 0)

                if score >= 8:
                    print(f"✅ Final implementation score {score}/10")
                    return current_code

                correction_prompt = f"""Fix the code based on issues and improvements suggested.

Plan:
{plan}
Code:
{current_code}
Issues: {critique.get('issues', [])}
Fixes: {critique.get('fixes', [])}
Return only the corrected code.
"""
                current_code = safe_initiate_chat(self.agents["corrector"], correction_prompt, self.user_proxy)

            except Exception:
                fallback = f"Improve the following code for task: {task}\nCode: {current_code}"
                current_code = safe_initiate_chat(self.agents["corrector"], fallback, self.user_proxy)

        return current_code


class LLMTaskAnalyzer:
    def __init__(self, agents):
        self.agent = agents["task_analyzer"]
        self.user_proxy = agents["user_proxy"]

    def analyze_task(self, task: str) -> dict:
        prompt = f"""Analyze this task and recommend one reasoning strategy: 
CODE_FIRST | PSEUDOCODE_FIRST | NEURO_SYMBOLIC. Return JSON:
{{
  "reasoning_strategy": "...",
  "complexity": X,
  "explanation": "..."
}}

Task: {task}
"""
        try:
            response = safe_initiate_chat(self.agent, prompt, self.user_proxy)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception:
            return {"reasoning_strategy": "CODE_FIRST", "complexity": 3, "explanation": "Default fallback"}
