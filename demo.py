#!/usr/bin/env python3
"""
Demo script for the Problem Solving Agent using LangChain and Ollama.
"""

from psa_agent import solve_problem

def main():
    print("Problem Solving Agent Demo")
    print("=" * 40)

    # Example problems
    problems = [
        "Write a Python function that computes the factorial of a number.",
        "Create a function to check if a string is a palindrome.",
        "Implement a simple calculator that can add, subtract, multiply, and divide."
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {problem}")
        print("-" * 50)

        try:
            result = solve_problem(problem)
            print(f"Solution:\n{result}")
        except Exception as e:
            print(f"Error solving problem: {e}")

        print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
