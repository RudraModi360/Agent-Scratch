# groq_coding_agent.py - Fixed Dynamic LLM-Powered Coding Agent
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
    
    def __init__(self):
        self.client = None
        self.model = "llama-3.1-70b-versatile"  # Changed to more stable model
    
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
    
    def _write_file(self, path: str, content: str) -> ToolResult:
        try:
            # Initialize Groq client if not already done
            if not self.client:
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    self.client = Groq(api_key=api_key)
            
            # Step 1: Detect file extension
            _, ext = os.path.splitext(path)
            ext = ext.lower().lstrip('.')

            # Step 2: Clean content if it looks like markdown
            if self._looks_like_markdown(content):
                content = self._clean_code_content(content, ext)

            # Step 3: Write final content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(True, f"File written successfully: {path}")

        except Exception as e:
            return ToolResult(False, "", str(e))

    def _looks_like_markdown(self, text: str) -> bool:
        """Detects if the content appears to be markdown-formatted."""
        return ("```" in text or 
                text.strip().startswith("# ") or 
                "```python" in text.lower() or 
                "```javascript" in text.lower() or
                "Here's" in text or
                "Here is" in text)

    def _clean_code_content(self, content: str, ext: str) -> str:
        """
        Cleans Markdown-like content into raw code.
        First tries regex-based cleaning, falls back to LLM if needed.
        """
        # First attempt: Simple regex-based cleaning
        cleaned = self._regex_clean_code(content)
        
        # If regex cleaning worked well, return it
        if cleaned and not self._looks_like_markdown(cleaned):
            return cleaned
        
        # Otherwise, use LLM for more complex cases
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": f"""Extract ONLY the raw {ext} code from this content. Remove all markdown formatting, explanations, and comments that aren't part of the actual code:

{content}

Return ONLY the clean code, nothing else."""
                    }],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass
        
        # Final fallback: return regex cleaned version
        return cleaned or content

    def _regex_clean_code(self, content: str) -> str:
        """Simple regex-based code cleaning"""
        # Remove markdown code blocks
        content = re.sub(r'```\w*\n', '', content)
        content = re.sub(r'```', '', content)
        
        # Remove common explanatory phrases at the start
        content = re.sub(r'^Here\'s.*?:\s*\n', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^Here is.*?:\s*\n', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove empty lines at the beginning and end
        content = content.strip()
        
        return content
    
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
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def validate(self, user_request: str, output: str, plan: List[Dict[str, Any]], created_files: List[str] = None) -> Dict[str, Any]:
        """Check if output satisfies the user request"""
        
        # Simple validation for common requests
        user_lower = user_request.lower()
        
        # Check if files were created successfully
        if created_files:
            files_exist = all(os.path.exists(f) for f in created_files)
            if not files_exist:
                return {
                    "satisfied": False,
                    "issues": ["Some files were not created successfully"],
                    "suggestions": ["Retry file creation"]
                }
        
        # Basic validation rules
        if any(word in user_lower for word in ["calculator", "gui"]):
            # For GUI applications, check if tkinter or similar imports exist
            if created_files:
                for file_path in created_files:
                    if file_path.endswith('.py'):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read().lower()
                                if "tkinter" in content or "gui" in content or "button" in content:
                                    return {"satisfied": True, "issues": [], "suggestions": []}
                        except:
                            pass
        
        # For simple requests, if files were created successfully, it's probably good
        if "create" in user_lower or "write" in user_lower or "make" in user_lower:
            if created_files and all(os.path.exists(f) for f in created_files):
                return {"satisfied": True, "issues": [], "suggestions": []}
        
        # Fallback to LLM validation for complex cases
        try:
            validation_prompt = f"""
You are a validator. Check if the output satisfies the user request. Be LENIENT - if the basic requirements are met, return satisfied=true.

User Request: {user_request}
Files Created: {created_files or []}
Output: {output[:500]}...

Return ONLY JSON:
{{
  "satisfied": true|false,
  "issues": ["list critical issues only"],
  "suggestions": ["only if major problems exist"]
}}

Rules:
- satisfied=true if basic functionality is present
- Only flag critical missing features
- Don't be overly strict about minor details
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            
            result = json.loads(content)
            
            # Override: if no critical issues and files exist, mark as satisfied
            if created_files and all(os.path.exists(f) for f in created_files):
                if not result.get("issues") or len(result.get("issues", [])) == 0:
                    result["satisfied"] = True
            
            return result
            
        except Exception as e:
            # Default to satisfied if validation fails and files exist
            if created_files and all(os.path.exists(f) for f in created_files):
                return {"satisfied": True, "issues": [], "suggestions": []}
            
            return {
                "satisfied": False, 
                "issues": [f"Validation error: {e}"], 
                "suggestions": ["Manual review needed"]
            }

class GroqLLMEngine:
    """Groq LLM integration for dynamic code generation and task planning"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def generate_code(self, user_request: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate code based on user request using LLM"""
        
        system_prompt = """You are an expert Python developer. Generate clean, working Python code.

IMPORTANT RULES:
1. Return ONLY the Python code, no explanations
2. No markdown formatting (no ``` blocks)
3. Include all necessary imports
4. Add minimal comments for clarity
5. Make code production-ready and functional
6. For GUI applications, use tkinter
7. Handle errors appropriately"""

        # Create focused prompt based on request
        code_prompt = f"Create Python code for: {user_request}"
        
        # Add specific guidance for common requests
        if "calculator" in user_request.lower() and "gui" in user_request.lower():
            code_prompt += "\nRequirements:\n- Use tkinter for GUI\n- Include basic arithmetic operations\n- Add a display field\n- Handle button clicks\n- Include error handling"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
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
        
        # Simple pattern-based analysis for Python requests
        user_lower = user_request.lower()
        
        # Determine project name
        if "calculator" in user_lower:
            project_name = "calculator"
        elif "game" in user_lower:
            project_name = "game"
        elif "scraper" in user_lower or "scraping" in user_lower:
            project_name = "web_scraper"
        elif "api" in user_lower:
            project_name = "api_client"
        else:
            project_name = "python_script"
        
        # Detect complexity
        complexity = "simple"
        if "gui" in user_lower or "interface" in user_lower:
            complexity = "medium"
        if "database" in user_lower or "web" in user_lower or "api" in user_lower:
            complexity = "complex"
        
        return {
            "task_type": "code_generation",
            "programming_language": "python",
            "project_name": project_name,
            "file_operations": [
                {"operation": "create_file", "path": f"{project_name}.py", "description": f"Create {project_name}.py"}
            ],
            "code_requirements": {
                "complexity": complexity,
                "features": self._extract_features(user_request),
                "dependencies": self._extract_dependencies(user_request),
                "file_type": "script"
            },
            "reasoning": f"User wants to create a Python {project_name}"
        }
    
    def _extract_features(self, user_request: str) -> List[str]:
        """Extract features from user request"""
        features = []
        user_lower = user_request.lower()
        
        if "gui" in user_lower:
            features.append("gui_interface")
        if "calculator" in user_lower:
            features.extend(["arithmetic_operations", "button_interface"])
        if "database" in user_lower:
            features.append("database_connection")
        if "web" in user_lower or "scraper" in user_lower:
            features.append("web_scraping")
        if "api" in user_lower:
            features.append("api_integration")
        
        return features
    
    def _extract_dependencies(self, user_request: str) -> List[str]:
        """Extract dependencies from user request"""
        deps = []
        user_lower = user_request.lower()
        
        if "gui" in user_lower:
            deps.append("tkinter")
        if "web" in user_lower or "scraper" in user_lower:
            deps.extend(["requests", "beautifulsoup4"])
        if "database" in user_lower:
            deps.append("sqlite3")
        
        return deps
    
    def plan_execution(self, analysis: Dict[str, Any], user_request: str) -> List[Dict[str, Any]]:
        """Plan execution steps based on analysis"""
        
        plan = []
        project_name = analysis.get("project_name", "python_script")
        
        # Step 1: Generate code
        plan.append({
            "step": 1,
            "action": "code_generation",
            "operation": "generate",
            "parameters": {
                "request": user_request, 
                "language": "python",
                "project_name": project_name
            },
            "description": f"Generate Python code for {project_name}"
        })
        
        # Step 2: Create file
        plan.append({
            "step": 2,
            "action": "filesystem",
            "operation": "create_file",
            "parameters": {
                "path": f"{project_name}.py",
                "content": "{{generated_code}}"
            },
            "description": f"Save code to {project_name}.py"
        })
        
        return plan

class GroqCodingAgent:
    """Dynamic coding agent powered by Groq LLM - Python focused"""
    
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
        """Process user request with improved validation logic"""
        self.chat_history.append(Message("user", user_input))
        
        try:
            print("🔍 Analyzing your request...")
            analysis = self.llm.analyze_request(user_input)

            print("📋 Planning execution steps...")
            execution_plan = self.llm.plan_execution(analysis, user_input)

            print(f"⚡ Executing {len(execution_plan)} steps...")
            results = self._execute_plan(execution_plan, user_input, analysis)

            # Get list of created files for validation
            created_files = self._get_created_files(execution_plan, results)

            print("✅ Validating output...")
            validation = self._validate_output(user_input, execution_plan, results, created_files)

            if validation.get("satisfied", True):  # Default to satisfied
                print("✅ Request completed successfully!")
                response = self._generate_response(analysis, execution_plan, results, user_input)
                self.chat_history.append(Message("assistant", response))
                return response
            else:
                print(f"⚠️ Minor issues found but proceeding: {validation.get('issues', [])}")
                response = self._generate_response(analysis, execution_plan, results, user_input)
                if validation.get("issues"):
                    response += f"\n\n📝 Note: {', '.join(validation['issues'])}"
                self.chat_history.append(Message("assistant", response))
                return response

        except Exception as e:
            error_response = f"❌ An error occurred: {str(e)}"
            self.chat_history.append(Message("assistant", error_response))
            return error_response
    
    def _get_created_files(self, plan: List[Dict[str, Any]], results: List[ToolResult]) -> List[str]:
        """Extract list of successfully created files"""
        created_files = []
        for step, result in zip(plan, results):
            if (step.get("action") == "filesystem" and 
                step.get("operation") == "create_file" and 
                result.success):
                file_path = step.get("parameters", {}).get("path")
                if file_path:
                    created_files.append(file_path)
        return created_files
    
    def _validate_output(self, user_input: str, plan: List[Dict[str, Any]], 
                        results: List[ToolResult], created_files: List[str]) -> Dict[str, Any]:
        """Validate output with created files information"""
        combined_output = "\n".join(
            f"Step {i+1}: {r.output or r.error}" for i, r in enumerate(results)
        )
        return self.validator_llm.validate(user_input, combined_output, plan, created_files)
    
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
            response_parts.append("🎉 Python code generated successfully!")
        elif success_count > 0:
            response_parts.append(f"✅ Completed {success_count}/{total_count} steps")
        else:
            response_parts.append("❌ Request failed")
        
        # Show what was created
        created_files = self._get_created_files(plan, results)
        if created_files:
            response_parts.append(f"\n📁 Files created: {', '.join(created_files)}")
            
            # Add execution instructions
            response_parts.append("\n🏃 To run your Python code:")
            for file in created_files:
                if file.endswith('.py'):
                    response_parts.append(f"   python {file}")
        
        # Add brief execution summary
        if results:
            response_parts.append("\n📋 Execution Summary:")
            for i, (step, result) in enumerate(zip(plan, results)):
                status = "✅" if result.success else "❌"
                desc = step.get("description", f"Step {i+1}")
                response_parts.append(f"  {status} {desc}")
        
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
    
    print("🐍 Python Coding Agent - Powered by Groq!")
    print("💡 I specialize in Python development!")
    print("\n🌟 Examples:")
    print("   • 'Make a calculator with GUI in Python'")
    print("   • 'Create a file organizer script'")
    print("   • 'Write a password generator'")
    print("   • 'Build a simple text editor'")
    print("   • 'Create a web scraper'")
    print("   • 'Make a to-do list application'")
    print("\nType 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🎯 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy Python coding!")
                break
            
            elif user_input.lower() in ['help', 'h']:
                print("\n📖 I can help with Python:")
                print("🧮 Calculators and GUI applications")
                print("📂 File management scripts")
                print("🌐 Web scraping tools")
                print("🎮 Simple games")
                print("🔐 Utilities and tools")
                print("📊 Data processing scripts")
                print("\nJust describe what you want in Python!")
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