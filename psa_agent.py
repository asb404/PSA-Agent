from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os
import tempfile
import sys
from runner import run_pytest_in_sandbox

llm = OllamaLLM(model="gemma3:4b")

def run_code_tests(code: str) -> str:
    """
    Run tests on generated code.
    Creates a temporary directory with code and tests, runs pytest in sandbox.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "solution.py")
        with open(code_file, "w") as f:
            f.write(code)

        test_file = os.path.join(tmpdir, "test_solution.py")
        with open(test_file, "w") as f:
            f.write("""
import pytest
from solution import *

def test_example():
    # Placeholder test - agent should generate appropriate tests
    assert True
""")

        result = run_pytest_in_sandbox(tmpdir, timeout=30, cpu_time=5, memory_bytes=100_000_000)
        return f"Return code: {result['returncode']}\nStdout: {result['stdout']}\nStderr: {result['stderr']}\nDuration: {result['duration']:.2f}s"

def solve_problem(problem: str) -> str:
    """
    Solve a problem using the LLM.
    Generates code and tests it.
    """
    code_prompt = PromptTemplate(
        input_variables=["problem"],
        template="""
You are a Python code generator. Given a problem, write Python code to solve it.

Problem: {problem}

Provide only the Python code, no explanations.
"""
    )

    generated_code = llm.invoke(code_prompt.format(problem=problem)).strip()

    if generated_code.startswith("```python"):
        generated_code = generated_code[9:].strip()
    if generated_code.startswith("```"):
        generated_code = generated_code[3:].strip()
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3].strip()

    test_result = run_code_tests(generated_code)

    analysis_prompt = PromptTemplate(
        input_variables=["problem", "code", "test_result"],
        template="""
You are a code reviewer. Analyze the code and test results for the problem.

Problem: {problem}
Code:
{code}
Test Result:
{test_result}

If the tests pass, say the solution is good.
If tests fail, suggest improvements.
"""
    )

    analysis = llm.invoke(analysis_prompt.format(problem=problem, code=generated_code, test_result=test_result))

    return f"Generated Code:\n{generated_code}\n\nTest Results:\n{test_result}\n\nAnalysis:\n{analysis}"

if __name__ == "__main__":
    problem = "Write a function to add two numbers."
    result = solve_problem(problem)
    print(result)
