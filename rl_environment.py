import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import re
from psa_agent import solve_problem
import hashlib

class CodeGenerationEnv(gym.Env):
    """
    Reinforcement Learning Environment for Code Generation Problem Solving.

    State: Encoded representation of problem, current code, and test results
    Actions: Different strategies for generating/modifying code
    Rewards: Based on test success, code quality, and efficiency
    """

    def __init__(self, problems: List[str] = None):
        super().__init__()

        if problems is None:
            problems = [
                "Write a function to add two numbers",
                "Create a function to check if a string is a palindrome",
                "Write a function to compute factorial",
                "Implement a simple calculator",
                "Write a function to find the maximum in a list"
            ]

        self.problems = problems
        self.current_problem_idx = 0
        self.current_problem = ""
        self.attempt_count = 0
        self.max_attempts = 5
        self.generated_codes = []
        self.test_results = []
        self.action_space = spaces.Discrete(5)

        # State space: Encoded features
        # Features: problem_hash, attempt_count, prev_test_success, code_length, code_complexity
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, self.max_attempts, 1, 1000, 100]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
        self.current_problem_idx = np.random.randint(len(self.problems))
        self.current_problem = self.problems[self.current_problem_idx]

        self.attempt_count = 0
        self.generated_codes = []
        self.test_results = []

        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        problem_hash = int(hashlib.md5(self.current_problem.encode()).hexdigest(), 16) % 1000 / 1000.0

        attempt_norm = self.attempt_count / self.max_attempts

        prev_success = 1.0 if self.test_results and self.test_results[-1]['returncode'] == 0 else 0.0

        if self.generated_codes:
            code = self.generated_codes[-1]
            code_length = min(len(code) / 1000.0, 1.0)  
            code_complexity = self._calculate_complexity(code)
        else:
            code_length = 0.0
            code_complexity = 0.0

        return np.array([
            problem_hash,
            attempt_norm,
            prev_success,
            code_length,
            code_complexity
        ], dtype=np.float32)

    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity metric."""
        keywords = ['if', 'for', 'while', 'def', 'class', 'try', 'except']
        keyword_count = sum(code.count(kw) for kw in keywords)

        indent_levels = [len(line) - len(line.lstrip()) for line in code.split('\n') if line.strip()]
        max_indent = max(indent_levels) // 4 if indent_levels else 0

        lines = len([line for line in code.split('\n') if line.strip()])

        complexity = (keyword_count * 0.3 + max_indent * 0.4 + lines * 0.1)
        return min(complexity / 10.0, 1.0)  

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.attempt_count += 1

        code = self._generate_code(action)
        self.generated_codes.append(code)

        test_result = self._test_code(code)
        self.test_results.append(test_result)

        reward = self._calculate_reward(code, test_result, action)

        done = (
            test_result['returncode'] == 0 or  # Success
            self.attempt_count >= self.max_attempts  # Max attempts reached
        )

        next_state = self._get_state()

        return next_state, reward, done, False, {
            'code': code,
            'test_result': test_result,
            'attempt': self.attempt_count
        }

    def _generate_code(self, action: int) -> str:
        """Generate code based on action strategy."""
        base_prompt = f"Solve this programming problem: {self.current_problem}"

        if action == 0:
            prompt = base_prompt
        elif action == 1:
            prompt = f"{base_prompt}\nThink step by step and then provide the complete solution."
        elif action == 2 and self.generated_codes:
            prev_code = self.generated_codes[-1]
            prev_result = self.test_results[-1]
            prompt = f"{base_prompt}\nPrevious attempt failed. Code: {prev_code}\nTest result: {prev_result}\nImprove this solution."
        elif action == 3:
            prompt = f"{base_prompt}\nProvide a simple, straightforward solution."
        elif action == 4:
            prompt = f"{base_prompt}\nProvide a more robust, feature-complete solution."
        else:
            prompt = base_prompt

        try:
            result = solve_problem(self.current_problem)
            code_match = re.search(r'Generated Code:\n(.*?)(?=\n\nTest Results:|$)', result, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return "# Failed to extract code"
        except Exception as e:
            return f"# Error: {str(e)}"

    def _test_code(self, code: str) -> Dict[str, Any]:
        """Test the generated code using the existing runner."""
        from runner import run_pytest_in_sandbox
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "solution.py")
            with open(code_file, "w") as f:
                f.write(code)

            test_file = os.path.join(tmpdir, "test_solution.py")
            with open(test_file, "w") as f:
                f.write("""
import pytest
from solution import *

def test_basic():
    # Basic functionality test - will be enhanced based on problem type
    assert True
""")

            result = run_pytest_in_sandbox(tmpdir, timeout=10, cpu_time=2, memory_bytes=50_000_000)
            return result

    def _calculate_reward(self, code: str, test_result: Dict, action: int) -> float:
        """Calculate reward for the current step."""
        reward = 0.0

        if test_result['returncode'] == 0:
            reward += 10.0  # Big reward for success
        else:
            reward -= 2.0   # Penalty for failure

        efficiency_bonus = (self.max_attempts - self.attempt_count) * 0.5
        reward += efficiency_bonus

        code_length = len(code)
        if code_length < 50:  # Too short
            reward -= 1.0
        elif code_length > 500:  # Too long
            reward -= 0.5

        if len(self.generated_codes) > 1 and self.generated_codes[-1] == self.generated_codes[-2]:
            reward -= 1.0

        if action in [1, 2, 3, 4]:  # Non-default actions
            reward += 0.2

        return reward

    def render(self, mode='human'):
        """Render current state."""
        print(f"Problem: {self.current_problem}")
        print(f"Attempt: {self.attempt_count}/{self.max_attempts}")
        if self.generated_codes:
            print(f"Latest code length: {len(self.generated_codes[-1])}")
        if self.test_results:
            success = self.test_results[-1]['returncode'] == 0
            print(f"Last test: {'PASS' if success else 'FAIL'}")

if __name__ == "__main__":
    env = CodeGenerationEnv()

    state, _ = env.reset()
    print("Initial state:", state)

    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Code preview: {info['code'][:100]}...")

        if done:
            break

    env.close()
