# groq_coding_agent.py - Dynamic LLM-Powered Coding Agent
import os
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess
from groq import Groq

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None

@dataclass
class Message:
    """Chat message structure"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[str] = None

@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    reasoning: Optional[str] = None
    confidence: float = 1.0

class BaseTool(ABC):
    """Base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class FileSystemTool(BaseTool):
    """File system operations tool"""
    
    @property
    def name(self) -> str:
        return "filesystem"
    
    @property
    def description(self) -> str:
        return "Create, read, write, delete files and directories"
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute file system operations"""
        try:
            if operation == "create_file":
                return self._create_file(kwargs.get('path'), kwargs.get('content', ''))
            elif operation == "create_directory":
                return self._create_directory(kwargs.get('path'))
            elif operation == "read_file":
                return self._read_file(kwargs.get('path'))
            elif operation == "write_file":
                return self._write_file(kwargs.get('path'), kwargs.get('content'))
            elif operation == "list_directory":
                return self._list_directory(kwargs.get('path', '.'))
            elif operation == "delete_file" or operation=="remove_file":
                return self._delete_file(kwargs.get('path'))
            elif operation == "file_exists":
                return self._file_exists(kwargs.get('path'))
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_file(self, path: str, content: str = '') -> ToolResult:
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(True, f"File created: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_directory(self, path: str) -> ToolResult:
        try:
            os.makedirs(path, exist_ok=True)
            return ToolResult(True, f"Directory created: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _read_file(self, path: str) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(True, f"File content read successfully", data={"content": content, "path": path})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    import os

    def _write_file(self, path: str, content: str) -> ToolResult:
        try:
            # Step 1: Detect file extension
            _, ext = os.path.splitext(path)
            ext = ext.lower().lstrip('.')

            # Step 2: Decide if content needs cleaning
            if self._looks_like_markdown(content):
                # Delegate to a cleaning function (could be LLM-powered or regex-based)
                content = self._clean_code_content(content, ext)

            # Step 3: Write final content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(True, f"Processed content written to: {path}")

        except Exception as e:
            return ToolResult(False, "", str(e))

    def _looks_like_markdown(self, text: str) -> bool:
        """Detects if the content appears to be markdown-formatted."""
        return "```" in text or text.strip().startswith("# ")

    def _clean_code_content(self, content: str, ext: str) -> str:
        """
        Cleans Markdown-like content into raw code.
        This can be replaced with an LLM call for smarter extraction.
        """
        self.client = Groq(api_key='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')
        self.model = "openai/gpt-oss-120b"
        cleaned=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": """
                           You are a strict code extractor.
            <CONTEXT>
            content
            </CONTEXT>

            You have given:
            1. A target file type (e.g., `.py`, `.js`, `.java`, `.html`, etc.).
            2. A text that may contain explanations, Markdown formatting, and one or more code blocks.

            Your job:
            - Return ONLY the raw code suitable for saving into the file.
            - Remove all explanations, comments unrelated to the actual program logic, and any Markdown formatting such as ``` fences.
            - If multiple code blocks are present, merge them in correct order as they would appear in a working file.
            - Preserve all indentation, spacing, and syntax exactly as in the code.
            - If the content already contains only code, return it as-is.
            - Never add extra explanations, text, or notes.

            Format:
            - Output ONLY the cleaned code, with no markdown and no extra commentary.

            Example:

            ---
            File type: `.py`
            Input:
            Here’s your Python code:

            ```python
            def hello():
                print("Hello World")
                
            ```

            Output:
            def hello():
            print("Hello World")

                           """}],
                temperature=0.1
            )
        return cleaned
    
    def _list_directory(self, path: str) -> ToolResult:
        try:
            items = []
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                item_type = "directory" if os.path.isdir(full_path) else "file"
                items.append({"name": item, "type": item_type})
            
            return ToolResult(True, f"Listed {len(items)} items", data={"items": items, "path": path})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _delete_file(self, path: str) -> ToolResult:
        try:
            if os.path.isfile(path):
                os.remove(path)
                return ToolResult(True, f"File deleted: {path}")
            elif os.path.isdir(path):
                os.rmdir(path)
                return ToolResult(True, f"Directory deleted: {path}")
            else:
                return ToolResult(False, "", f"Path not found: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _file_exists(self, path: str) -> ToolResult:
        try:
            exists = os.path.exists(path)
            return ToolResult(True, f"Path {'exists' if exists else 'does not exist'}", 
                            data={"exists": exists, "path": path})
        except Exception as e:
            return ToolResult(False, "", str(e))

class CommandTool(BaseTool):
    """Execute system commands"""
    
    @property
    def name(self) -> str:
        return "command"
    
    @property
    def description(self) -> str:
        return "Execute system commands and shell operations"
    
    def execute(self, command: str, **kwargs) -> ToolResult:
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=kwargs.get('working_directory', '.'),
                timeout=kwargs.get('timeout', 30)
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            
            success = result.returncode == 0
            return ToolResult(success, output, None if success else result.stderr)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Command timed out")
        except Exception as e:
            return ToolResult(False, "", str(e))

class ValidatorLLMEngine:
    """LLM that validates output against user intent"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def validate(self, user_request: str, output: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if output satisfies the user request"""
        validation_prompt = f"""
You are a strict validator. Compare the original user request with the provided output and execution plan.
Return ONLY a JSON object with:
{{
  "satisfied": true|false,
  "issues": ["list issues if any"],
  "suggestions": ["list improvements or corrections if any"]
}}

User Request:
{user_request}

Execution Plan:
{json.dumps(plan, indent=2)}

Generated Output:
{output}

Rules:
- satisfied = true ONLY if the output fully matches the request's requirements.
- If there is any missing functionality, wrong format, or inaccuracy, set satisfied = false and list them in issues.
- suggestions must be clear, actionable improvements.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            return json.loads(content)
        except Exception as e:
            return {"satisfied": False, "issues": [f"Validator error: {e}"], "suggestions": []}

class GroqLLMEngine:
    """Groq LLM integration for dynamic code generation and task planning"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def generate_code(self, user_request: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate code based on user request using LLM"""
        
        system_prompt = """You are an expert software developer and coding assistant. 
Generate high-quality, working code based on user requests. Consider:

1. Best practices for the language
2. Proper error handling
3. Clean, readable code structure
4. Appropriate comments and documentation
5. Security considerations
6. Efficiency and performance

Provide ONLY the code without explanations unless specifically asked.
Make the code production-ready and functional."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        # Add context if provided
        if context:
            context_str = f"Additional context: {json.dumps(context, indent=2)}"
            messages.append({"role": "user", "content": context_str})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return LLMResponse(content=content, confidence=0.9)
            
        except Exception as e:
            return LLMResponse(
                content=f"# Error generating code: {str(e)}\n# Please check your request and try again.",
                confidence=0.0
            )
    
    def analyze_request(self, user_request: str) -> Dict[str, Any]:
        """Analyze user request to determine what needs to be done"""
        
        analysis_prompt = f"""Analyze this user request and return a JSON response with the following structure:
{{
    "task_type": "code_generation|file_operation|project_setup|help|other",
    "programming_language": "python|javascript|java|cpp|etc or null",
    "project_name": "suggested_project_name or null",
    "file_operations": [
        {{"operation": "create_file|read_file|delete_file|list_files", "path": "file_path", "description": "what to do"}}
    ],
    "code_requirements": {{
        "complexity": "simple|medium|complex",
        "features": ["feature1", "feature2"],
        "dependencies": ["dep1", "dep2"],
        "file_type": "script|class|module|application"
    }},
    "reasoning": "explanation of what the user wants"
}}

User request: {user_request}

Respond with ONLY valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's valid JSON
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            # Fallback analysis
            return {
                "task_type": "code_generation" if any(word in user_request.lower() 
                           for word in ["create", "write", "generate", "make", "build"]) else "other",
                "programming_language": self._detect_language_fallback(user_request),
                "project_name": "my_project",
                "file_operations": [],
                "code_requirements": {
                    "complexity": "simple",
                    "features": [],
                    "dependencies": [],
                    "file_type": "script"
                },
                "reasoning": f"Fallback analysis due to JSON parse error: {str(e)}"
            }
        except Exception as e:
            return {
                "task_type": "other",
                "programming_language": None,
                "project_name": None,
                "file_operations": [],
                "code_requirements": {"complexity": "simple", "features": [], "dependencies": [], "file_type": "script"},
                "reasoning": f"Analysis failed: {str(e)}"
            }

    
    def _detect_language_fallback(self, user_request: str) -> str:
        """Fallback language detection"""
        languages = {
            'python': ['python', 'py', 'django', 'flask'],
            'javascript': ['javascript', 'js', 'node', 'react'],
            'java': ['java', 'spring'],
            'cpp': ['c++', 'cpp'],
            'c': [' c ', 'c language'],
            'html': ['html', 'web'],
            'css': ['css', 'style']
        }
        
        user_request_lower = user_request.lower()
        for lang, keywords in languages.items():
            if any(keyword in user_request_lower for keyword in keywords):
                return lang
        
        return 'python'  # Default
    
    def plan_execution(self, analysis: Dict[str, Any], user_request: str) -> List[Dict[str, Any]]:
        """Plan execution steps based on analysis"""
        
        planning_prompt = f"""Based on this analysis of a user request, create an execution plan.
Return a JSON array of steps to execute:

Analysis: {json.dumps(analysis, indent=2)}
Original request: {user_request}

Return format:
[
    {{
        "step": 1,
        "action": "filesystem|command|code_generation",
        "operation": "specific_operation",
        "parameters": {{"key": "value"}},
        "description": "what this step does"
    }}
]

Consider:
- Creating necessary directories first
- Generating code before writing files
- Setting up project structure if needed
- Installing dependencies if required

Respond with ONLY valid JSON array."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            return json.loads(content)
            
        except Exception as e:
            # Fallback execution plan
            return self._create_fallback_plan(analysis, user_request)
    
    def _create_fallback_plan(self, analysis: Dict[str, Any], user_request: str) -> List[Dict[str, Any]]:
        """Create a fallback execution plan"""
        plan = []
        
        if analysis.get("task_type") == "code_generation":
            # Simple code generation plan
            plan.append({
                "step": 1,
                "action": "code_generation",
                "operation": "generate",
                "parameters": {"request": user_request, "language": analysis.get("programming_language", "python")},
                "description": "Generate code based on request"
            })
            
            # Determine file name
            lang = analysis.get("programming_language", "python")
            extensions = {"python": ".py", "javascript": ".js", "java": ".java", "cpp": ".cpp"}
            ext = extensions.get(lang, ".txt")
            filename = f"{analysis.get('project_name', 'generated_code')}{ext}"
            
            plan.append({
                "step": 2,
                "action": "filesystem",
                "operation": "create_file",
                "parameters": {"path": filename, "content": "{{generated_code}}"},
                "description": f"Save generated code to {filename}"
            })
        
        elif analysis.get("file_operations"):
            # File operations
            for i, op in enumerate(analysis["file_operations"]):
                plan.append({
                    "step": i + 1,
                    "action": "filesystem",
                    "operation": op["operation"],
                    "parameters": {"path": op.get("path", ""), "content": ""},
                    "description": op.get("description", f"Execute {op['operation']}")
                })
        
        return plan

class GroqCodingAgent:
    """Dynamic coding agent powered by Groq LLM"""
    
    def __init__(self, groq_api_key: str):
        # Initialize tools
        self.tools = {
            'filesystem': FileSystemTool(),
            'command': CommandTool()
        }
        
        # Initialize LLM engine
        self.llm = GroqLLMEngine(groq_api_key)
        self.validator_llm = ValidatorLLMEngine(groq_api_key)
        # State management
        self.chat_history: List[Message] = []
        self.current_directory = os.getcwd()
        self.context = {}
    
    def process_request(self, user_input: str) -> str:
        self.chat_history.append(Message("user", user_input))
        retries = 0
        max_retries = 5
        refined_input = user_input

        while retries < max_retries:
            try:
                print("🔍 Analyzing your request...")
                analysis = self.llm.analyze_request(refined_input)

                print("📋 Planning execution steps...")
                execution_plan = self.llm.plan_execution(analysis, refined_input)

                print(f"⚡ Executing {len(execution_plan)} steps...")
                results = self._execute_plan(execution_plan, refined_input, analysis)

                print("🧐 Validating output...")
                validation = self._validate_output(refined_input, execution_plan, results)

                if validation.get("satisfied", False):
                    print("✅ Output satisfies user intent")
                    response = self._generate_response(analysis, execution_plan, results, refined_input)
                    self.chat_history.append(Message("assistant", response))
                    return response
                else:
                    print(f"⚠️ Output did not pass validation. Issues: {validation['issues']}")
                    retries += 1
                    refined_input = refined_input + "\n\nRefinement: " + " ".join(validation.get("suggestions", []))

            except Exception as e:
                error_response = f"❌ An error occurred: {str(e)}"
                self.chat_history.append(Message("assistant", error_response))
                return error_response

        return "❌ Unable to satisfy user intent after multiple attempts."

    
    def _validate_output(self, user_input: str, plan: List[Dict[str, Any]], results: List[ToolResult]) -> Dict[str, Any]:
        """Send output to validator LLM for verification"""
        # Merge all tool outputs
        combined_output = "\n".join(
            f"Step {i+1}: {r.output or r.error}" for i, r in enumerate(results)
        )
        return self.validator_llm.validate(user_input, combined_output, plan)
    
    
    def _execute_plan(self, plan: List[Dict[str, Any]], user_request: str, analysis: Dict[str, Any]) -> List[ToolResult]:
        """Execute the planned steps"""
        results = []
        generated_code = None
        
        for step in plan:
            try:
                action = step.get("action")
                operation = step.get("operation")
                parameters = step.get("parameters", {})
                
                print(f"  📌 Step {step.get('step', '?')}: {step.get('description', 'Executing...')}")
                
                if action == "code_generation":
                    # Generate code using LLM
                    context = {
                        "language": parameters.get("language"),
                        "user_request": user_request,
                        "analysis": analysis
                    }
                    
                    llm_response = self.llm.generate_code(user_request, context)
                    generated_code = llm_response.content
                    
                    results.append(ToolResult(
                        success=True,
                        output=f"Code generated ({len(generated_code)} characters)",
                        data={"generated_code": generated_code}
                    ))
                
                elif action == "filesystem":
                    # Handle filesystem operations
                    if operation == "create_file" and "{{generated_code}}" in parameters.get("content", ""):
                        # Replace placeholder with actual generated code
                        parameters["content"] = generated_code or "# Generated code placeholder"
                    
                    result = self.tools['filesystem'].execute(operation, **parameters)
                    results.append(result)
                
                elif action == "command":
                    # Handle command execution
                    command = parameters.get("command", "")
                    result = self.tools['command'].execute(command, **parameters)
                    results.append(result)
                
                else:
                    results.append(ToolResult(False, "", f"Unknown action: {action}"))
            
            except Exception as e:
                results.append(ToolResult(False, "", f"Step execution failed: {str(e)}"))
        
        return results
    
    def _generate_response(self, analysis: Dict[str, Any], plan: List[Dict[str, Any]], 
                          results: List[ToolResult], user_request: str) -> str:
        """Generate human-readable response"""
        
        success_count = sum(1 for result in results if result.success)
        total_count = len(results)
        
        response_parts = []
        
        # Header
        if success_count == total_count and total_count > 0:
            response_parts.append("🎉 Request completed successfully!")
        elif success_count > 0:
            response_parts.append(f"✅ Completed {success_count}/{total_count} steps")
        else:
            response_parts.append("❌ Request failed")
        
        # Add analysis summary
        task_type = analysis.get("task_type", "unknown")
        if task_type != "unknown":
            response_parts.append(f"📊 Task type: {task_type.replace('_', ' ').title()}")
        
        language = analysis.get("programming_language")
        if language:
            response_parts.append(f"💻 Language: {language.title()}")
        
        # Show execution results
        if results:
            response_parts.append("\n📋 Execution Summary:")
            for i, (step, result) in enumerate(zip(plan, results)):
                status = "✅" if result.success else "❌"
                desc = step.get("description", f"Step {i+1}")
                response_parts.append(f"  {status} {desc}")
                
                if result.success and result.output:
                    response_parts.append(f"      └─ {result.output}")
                elif not result.success and result.error:
                    response_parts.append(f"      └─ Error: {result.error}")
        
        # Add helpful information
        if success_count > 0:
            # Check if any files were created
            created_files = []
            for step, result in zip(plan, results):
                if (step.get("action") == "filesystem" and 
                    step.get("operation") == "create_file" and 
                    result.success):
                    file_path = step.get("parameters", {}).get("path")
                    if file_path:
                        created_files.append(file_path)
            
            if created_files:
                response_parts.append(f"\n📁 Files created: {', '.join(created_files)}")
                
                # Add execution instructions for code files
                code_files = [f for f in created_files if f.endswith(('.py', '.js', '.java', '.cpp'))]
                if code_files:
                    response_parts.append("\n🏃 To run your code:")
                    for file in code_files:
                        if file.endswith('.py'):
                            response_parts.append(f"   python {file}")
                        elif file.endswith('.js'):
                            response_parts.append(f"   node {file}")
                        elif file.endswith('.java'):
                            response_parts.append(f"   javac {file} && java {file.replace('.java', '')}")
        
        return '\n'.join(response_parts)

# Example usage and configuration
def main():
    """Main function for testing the agent"""
    
    # Check for Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("   Get your API key from: https://console.groq.com/keys")
        return
    
    # Initialize the agent
    agent = GroqCodingAgent(groq_api_key)
    
    print("🤖 Groq-Powered Coding Agent Started!")
    print("💡 I can handle ANY coding task using AI intelligence!")
    print("\n🌟 Examples:")
    print("   • 'Write a simple file reader in Python'")
    print("   • 'Create a REST API client'")
    print("   • 'Make a sorting algorithm in JavaScript'")
    print("   • 'Build a calculator with GUI'")
    print("   • 'Create a web scraper for news'")
    print("   • 'Write a database connection script'")
    print("\nType 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🎯 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Happy coding!")
                break
            
            elif user_input.lower() in ['help', 'h']:
                print("\n📖 I can help with:")
                print("🔧 Any coding task in any language")
                print("📂 File operations (create, read, write, delete)")
                print("🗂️ Project setup and structure")
                print("🧩 Algorithm implementation")
                print("🌐 API clients and web scrapers")
                print("💾 Database scripts")
                print("🎮 Games and applications")
                print("\nJust describe what you want in natural language!")
                continue
            
            elif user_input.lower() in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if user_input:
                response = agent.process_request(user_input)
                print(f"\n🤖 Agent:\n{response}")
                print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Agent interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()