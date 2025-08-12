import unittest
import os
import shutil
import json
from testing_agent import GeneralGroqAgent

RESULTS_FILE = 'results.txt'

def log_result(test_name, plan, output):
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n=== {test_name} ===\n")
        f.write("Plan:\n")
        f.write(json.dumps(plan, indent=2))
        f.write("\nOutput:\n")
        if isinstance(output, dict):
            f.write(json.dumps(output, indent=2))
        else:
            f.write(str(output))
        f.write("\n====================\n")

class TestAgentFileGeneration(unittest.TestCase):
    def setUp(self):
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set for tests")
        self.agent = GeneralGroqAgent(groq_api_key=groq_api_key)
        self.generated_files = [
            'generated_code.py',
            'creative_poem.txt',
            'factorial.py',
            'multi_step.txt',
        ]
        self.generated_dirs = ['test_dir']
        # Clear results.txt at start of test run
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)

    def tearDown(self):
        for f in self.generated_files:
            if os.path.exists(f):
                os.remove(f)
        for d in self.generated_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_code_generation(self):
        plan = [
            {
                "step": 1,
                "action": "llm_generation",
                "operation": "generate_code",
                "parameters": {"request": "Write a Python function that returns the square of a number.", "language": "python"},
                "description": "Generate code"
            },
            {
                "step": 2,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": "generated_code.py", "content": "{{step_1.data.content}}"},
                "description": "Write code to file"
            }
        ]
        result = self.agent.execute_plan(plan)
        output = ""
        if os.path.exists('generated_code.py'):
            with open('generated_code.py') as f:
                output = f.read()
        log_result("test_code_generation", plan, output)
        self.assertIn('def', output)
        self.assertIn('return', output)
        self.assertNotIn('{{', output)

    def test_creative_writing(self):
        plan = [
            {
                "step": 1,
                "action": "llm_generation",
                "operation": "generate_response",
                "parameters": {"request": "Write a short poem about the sea."},
                "description": "Generate poem"
            },
            {
                "step": 2,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": "creative_poem.txt", "content": "{{step_1.data.content}}"},
                "description": "Write poem to file"
            }
        ]
        result = self.agent.execute_plan(plan)
        output = ""
        if os.path.exists('creative_poem.txt'):
            with open('creative_poem.txt') as f:
                output = f.read()
        log_result("test_creative_writing", plan, output)
        self.assertGreater(len(output.strip()), 10)
        self.assertNotIn('{{', output)

    def test_directory_creation(self):
        plan = [
            {
                "step": 1,
                "action": "filesystem",
                "operation": "create_directory",
                "parameters": {"path": "test_dir"},
                "description": "Create test directory"
            }
        ]
        result = self.agent.execute_plan(plan)
        output = {
            "exists": os.path.exists('test_dir'),
            "isdir": os.path.isdir('test_dir')
        }
        log_result("test_directory_creation", plan, output)
        self.assertTrue(output["exists"])
        self.assertTrue(output["isdir"])

    def test_multi_step_plan(self):
        plan = [
            {
                "step": 1,
                "action": "llm_generation",
                "operation": "generate_code",
                "parameters": {"request": "Write a Python function to compute factorial.", "language": "python"},
                "description": "Generate factorial code"
            },
            {
                "step": 2,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": "factorial.py", "content": "{{step_1.data.content}}"},
                "description": "Write code to file"
            },
            {
                "step": 3,
                "action": "filesystem",
                "operation": "read_file",
                "parameters": {"path": "factorial.py"},
                "description": "Read back code"
            },
            {
                "step": 4,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": "multi_step.txt", "content": "Code:\n{{step_2.data.content}}\nReadback:\n{{step_3.data.content}}"},
                "description": "Write summary file"
            }
        ]
        result = self.agent.execute_plan(plan)
        output = ""
        if os.path.exists('multi_step.txt'):
            with open('multi_step.txt') as f:
                output = f.read()
        log_result("test_multi_step_plan", plan, output)
        self.assertIn('def', output)
        self.assertIn('factorial', output)
        self.assertNotIn('{{', output)

    def test_placeholder_resolution(self):
        plan = [
            {
                "step": 1,
                "action": "llm_generation",
                "operation": "generate_response",
                "parameters": {"request": "Write a haiku about autumn."},
                "description": "Generate haiku"
            },
            {
                "step": 2,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": "creative_poem.txt", "content": "Haiku:\n{{step_1.data.content}}\nEnd."},
                "description": "Write haiku to file"
            }
        ]
        result = self.agent.execute_plan(plan)
        output = ""
        if os.path.exists('creative_poem.txt'):
            with open('creative_poem.txt') as f:
                output = f.read()
        log_result("test_placeholder_resolution", plan, output)
        self.assertIn('Haiku:', output)
        self.assertIn('End.', output)
        self.assertNotIn('{{', output)

if __name__ == '__main__':
    unittest.main()
