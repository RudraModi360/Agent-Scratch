import os
import json
import re
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union, Callable,Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import subprocess
from datetime import datetime
from groq import Groq
from enum import Enum
from pydantic import BaseModel

class TaskType(Enum):
    """Types of tasks the agent can handle"""
    CODE_GENERATION = "code_generation"
    FILE_MANAGEMENT = "file_management" 
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    AUTOMATION = "auomation"
    CREATIVE_WRITING = "creative_writing"
    PROBLEM_SOLVING = "problem_solving"
    SYSTEM_ADMINISTRATION = "system_administration"
    WEB_SCRAPING = "web_scraping"
    EDIT_FILE = "edit_file"
    API_INTEGRATION = "api_integration"
    WEB_SEARCH="web_search"
    GENERAL = "general"

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class Message:
    """Chat message structure"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    reasoning: Optional[str] = None
    confidence: float = 1.0
    tool_calls: Optional[List[Dict[str, Any]]] = None

@dataclass
class TaskContext:
    """Context for task execution"""
    task_type: TaskType
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    working_directory: str = "."

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
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        return True

class EditFileTool(BaseTool):
    """Edit existing files by exact text replacement"""
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return ("Modify EXISTING files by exact text replacement. Use this for files that already exist. "
                "MANDATORY: Always read_file first to see current content before editing. "
                "Text must match exactly including whitespace. "
                "Example: {\"file_path\": \"src/app.js\", \"old_text\": \"const x = 1;\", \"new_text\": \"const x = 2;\"}")

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to edit. For files in current directory use just filename (e.g. 'app.js'). For subdirectories use 'src/app.js'. DO NOT use absolute paths or leading slashes."
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace (must match perfectly including spaces/newlines)"
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                    "default": False
                }
            },
            "required": ["file_path", "old_text", "new_text"]
        }

    def execute(self, **kwargs) -> ToolResult:
        import os
        file_path = kwargs.get('file_path')
        old_text = kwargs.get('old_text')
        new_text = kwargs.get('new_text')
        replace_all = kwargs.get('replace_all', False)
        # Validate required parameters
        if not file_path or not isinstance(file_path, str):
            return ToolResult(False, "", "Missing or invalid 'file_path' parameter")
        if not old_text or not isinstance(old_text, str):
            return ToolResult(False, "", "Missing or invalid 'old_text' parameter")
        if new_text is None or not isinstance(new_text, str):
            return ToolResult(False, "", "Missing or invalid 'new_text' parameter")
        try:
            # Only allow relative paths, no leading slashes
            if os.path.isabs(file_path) or file_path.startswith("/"):
                return ToolResult(False, "", "Absolute paths or leading slashes are not allowed")
            resolved_path = os.path.abspath(file_path)
            if not os.path.exists(resolved_path):
                return ToolResult(False, "", f"File does not exist: {file_path}")
            # Read current content
            with open(resolved_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            # Perform the replacement
            if replace_all:
                updated_content = original_content.replace(old_text, new_text)
                replacement_count = original_content.count(old_text)
            else:
                updated_content = original_content.replace(old_text, new_text, 1)
                replacement_count = 1 if old_text in original_content else 0
            # Write the updated content
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            if replacement_count > 0:
                return ToolResult(True, f"Replaced {replacement_count} occurrence(s) in {file_path}")
            else:
                return ToolResult(False, "", f"No occurrences of '{old_text}' found in {file_path}")
        except Exception as e:
            return ToolResult(False, "", f"Error: Failed to edit file - {e}")
        
class FileSystemTool(BaseTool):
    """Enhanced file system operations tool"""
    
    @property
    def name(self) -> str:
        return "filesystem"
    
    @property
    def description(self) -> str:
        return "Create, read, write, delete, search files and directories. Supports text processing and file analysis."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string", 
                    "enum": ["create_file", "read_file", "write_file", "append_file", 
                           "delete_file", "create_directory", "list_directory", 
                           "search_files", "file_stats", "copy_file", "move_file"]
                },
                "path": {"type": "string"},
                "content": {"type": "string", "default": ""},
                "pattern": {"type": "string"},
                "recursive": {"type": "boolean", "default": False}
            },
            "required": ["operation"]
        }
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute file system operations"""
        try:
            operations = {
                "create_file": self._create_file,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "append_file": self._append_file,
                "delete_file": self._delete_file,
                "create_directory": self._create_directory,
                "list_directory": self._list_directory,
                "search_files": self._search_files,
                "file_stats": self._file_stats,
                "copy_file": self._copy_file,
                "move_file": self._move_file
            }
            if operation in operations:
                return operations[operation](**kwargs)
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_file(self, path: str, content: str = '', **kwargs) -> ToolResult:
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            stats = os.stat(path)
            return ToolResult(True, f"File created: {path}", 
                            data={"path": path, "size": stats.st_size})
        except Exception as e:
            return ToolResult(False, "", str(e))

    def _write_file(self, path: str, content: str = '', **kwargs) -> ToolResult:
        """Overwrite the content of a file at the given path."""
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            stats = os.stat(path)
            return ToolResult(True, f"File overwritten: {path}", data={"path": path, "size": stats.st_size})
        except Exception as e:
            return ToolResult(False, "", str(e))

    
    def _read_file(self, path: str, **kwargs) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            stats = os.stat(path)
            return ToolResult(True, "File read successfully", 
                            data={
                                "content": content, 
                                "path": path,
                                "size": stats.st_size,
                                "lines": len(content.splitlines())
                            })
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _search_files(self, pattern: str, path: str = ".", recursive: bool = True, **kwargs) -> ToolResult:
        try:
            import glob
            search_pattern = os.path.join(path, "**", pattern) if recursive else os.path.join(path, pattern)
            matches = glob.glob(search_pattern, recursive=recursive)
            
            files_info = []
            for match in matches:
                if os.path.isfile(match):
                    stats = os.stat(match)
                    files_info.append({
                        "path": match,
                        "size": stats.st_size,
                        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                    })
            
            return ToolResult(True, f"Found {len(files_info)} files", 
                            data={"matches": files_info, "pattern": pattern})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _file_stats(self, path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(False, "", f"Path does not exist: {path}")
            
            stats = os.stat(path)
            info = {
                "path": path,
                "size": stats.st_size,
                "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "is_file": os.path.isfile(path),
                "is_directory": os.path.isdir(path)
            }
            
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    info["lines"] = len(content.splitlines())
                    info["words"] = len(content.split())
                    info["characters"] = len(content)
            
            return ToolResult(True, "File stats retrieved", data=info)
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _append_file(self, path: str, content: str, **kwargs) -> ToolResult:
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(True, f"Content appended to {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _copy_file(self, source: str, destination: str, **kwargs) -> ToolResult:
        try:
            import shutil
            shutil.copy2(source, destination)
            return ToolResult(True, f"Copied {source} to {destination}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _move_file(self, source: str, destination: str, **kwargs) -> ToolResult:
        try:
            import shutil
            shutil.move(source, destination)
            return ToolResult(True, f"Moved {source} to {destination}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_directory(self, path: str, **kwargs) -> ToolResult:
        try:
            os.makedirs(path, exist_ok=True)
            return ToolResult(True, f"Directory created: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _list_directory(self, path: str = ".", **kwargs) -> ToolResult:
        try:
            items = []
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                stats = os.stat(full_path)
                item_info = {
                    "name": item,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "size": stats.st_size if os.path.isfile(full_path) else 0,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                }
                items.append(item_info)
            
            return ToolResult(True, f"Listed {len(items)} items", 
                            data={"items": items, "path": path})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _delete_file(self, path: str, **kwargs) -> ToolResult:
        try:
            if os.path.isfile(path):
                os.remove(path)
                return ToolResult(True, f"File deleted: {path}")
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                return ToolResult(True, f"Directory deleted: {path}")
            else:
                return ToolResult(False, "", f"Path not found: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))

class CommandTool(BaseTool):
    """Enhanced system command execution tool"""
    
    @property
    def name(self) -> str:
        return "command"
    
    @property
    def description(self) -> str:
        return "Execute system commands, shell scripts, and manage processes"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_directory": {"type": "string", "default": "."},
                "timeout": {"type": "integer", "default": 30},
                "capture_output": {"type": "boolean", "default": True},
                "shell": {"type": "boolean", "default": True}
            },
            "required": ["command"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            command = kwargs.get('command')
            if not command:
                return ToolResult(False, '', 'Missing required parameter: command')
            result = subprocess.run(
                command,
                shell=kwargs.get('shell', True),
                capture_output=kwargs.get('capture_output', True),
                text=True,
                cwd=kwargs.get('working_directory', '.'),
                timeout=kwargs.get('timeout', 30)
            )
            
            output = result.stdout
            error = result.stderr
            
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                data={
                    "return_code": result.returncode,
                    "command": command,
                    "stdout": output,
                    "stderr": error
                }
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Command timed out")
        except Exception as e:
            return ToolResult(False, "", str(e))

class WebTool(BaseTool):
    """Web operations tool for research and data gathering"""
    
    @property
    def name(self) -> str:
        return "web"
    
    @property
    def description(self) -> str:
        return "Perform web requests, download content, and basic web scraping"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "data": {"type": "object"},
                "headers": {"type": "object"},
                "timeout": {"type": "integer", "default": 30}
            },
            "required": ["data","description"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            data = kwargs.get("data", "Do the web search")
            description = kwargs.get("description", "Do the web search")

            # Build the user request string
            user_request = f"Search the web for: {data}, for additional context use Description : {description}"

            # Get Groq API key from environment
            from os import getenv
            groq_api_key = getenv('GROQ_API_KEY')
            if not groq_api_key:
                return ToolResult(False, '', 'Missing GROQ_API_KEY environment variable')

            # Initialize Groq LLM Engine
            engine = GroqLLMEngine(groq_api_key)

            # Perform agentic tool call
            llm_response = engine.agentic_tool_call(engine.client,user_request)

            # Return a successful ToolResult with details
            return ToolResult(
                success=True,
                output=llm_response.content,
                data={
                    "content":llm_response.content,
                    "reasoning": llm_response.reasoning,
                    "tool_calls": llm_response.tool_calls
                }
            )

        except Exception as e:
            return ToolResult(False, "", str(e))

class DataProcessingTool(BaseTool):
    """Data analysis and processing tool"""
    
    @property
    def name(self) -> str:
        return "data_processing"
    
    @property
    def description(self) -> str:
        return "Process, analyze, and manipulate data (JSON, CSV, text)"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string", 
                    "enum": ["parse_json", "parse_csv", "analyze_text", "transform_data", "statistics"]
                },
                "data": {"type": "string"},
                "format": {"type": "string", "enum": ["json", "csv", "text"]},
                "options": {"type": "object"}
            },
            "required": ["operation", "data"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            operation = kwargs.get('operation')
            data = kwargs.get('data')
            if not operation:
                return ToolResult(False, '', 'Missing required parameter: operation')
            if data is None:
                return ToolResult(False, '', 'Missing required parameter: data')
            if operation == "parse_json":
                parsed = json.loads(data)
                return ToolResult(True, "JSON parsed successfully", data={"parsed": parsed})
            
            elif operation == "analyze_text":
                lines = data.splitlines()
                words = data.split()
                chars = len(data)
                
                analysis = {
                    "lines": len(lines),
                    "words": len(words),
                    "characters": chars,
                    "average_line_length": chars / max(len(lines), 1),
                    "most_common_words": self._get_word_frequency(words)[:10]
                }
                
                return ToolResult(True, "Text analyzed", data=analysis)
            
            elif operation == "statistics":
                try:
                    numbers = [float(x) for x in data.split() if x.replace('.', '').replace('-', '').isdigit()]
                    if not numbers:
                        return ToolResult(False, "", "No numeric data found")
                    
                    stats = {
                        "count": len(numbers),
                        "sum": sum(numbers),
                        "mean": sum(numbers) / len(numbers),
                        "min": min(numbers),
                        "max": max(numbers),
                        "range": max(numbers) - min(numbers)
                    }
                    
                    return ToolResult(True, "Statistics calculated", data=stats)
                except Exception as e:
                    return ToolResult(False, "", f"Could not process numeric data: {str(e)}")
            
            else:
                return ToolResult(False, "", f"Unknown data operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _get_word_frequency(self, words: List[str]) -> List[tuple]:
        from collections import Counter
        # Clean words and count frequency
        clean_words = [word.lower().strip('.,!?";:') for word in words if len(word) > 2]
        return Counter(clean_words).most_common()

class GroqLLMEngine:
    """Enhanced Groq LLM integration for general task processing"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def analyze_request(self, user_request: str, available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze user request to determine task type and requirements"""
        
        system_prompt = """You are a task analysis expert. Analyze the user request and return a JSON response with:
1. task_type: The primary type of task
2. complexity: simple/medium/complex
3. required_tools: List of tools needed
4. subtasks: Break down into smaller tasks
5. estimated_steps: Number of execution steps
6. reasoning: Brief explanation

Available task types: code_generation, file_management, edit ,research, data_analysis, automation, creative_writing, problem_solving, system_administration, web_scraping, api_integration, general

Available tools: """ + str(available_tools or [])

        user_prompt = f"""
        Analyze this request: {user_request} ,
        You have following list of tools : {available_tools} ,
        Now by analysing the user request , pick the best suited [action,required_tools] for fulfilling this tasks.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                analysis = json.loads(content)
                
                # Ensure required fields exist
                analysis.setdefault("task_type", "general")
                analysis.setdefault("complexity", "medium")
                analysis.setdefault("required_tools", ["filesystem"])
                analysis.setdefault("subtasks", [])
                analysis.setdefault("estimated_steps", 1)
                analysis.setdefault("reasoning", "General task analysis")
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback to pattern-based analysis
                return self._fallback_analysis(user_request)
                
        except Exception as e:
            return self._fallback_analysis(user_request)
    
    def agentic_tool_call(self,client:Groq,user_input:str):
        client = Groq()

        response = client.chat.completions.create(
            model="compound-beta",
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
        return response.choices[0].message
    
    def _fallback_analysis(self, user_request: str) -> Dict[str, Any]:
        """Semantic analysis for file editing/formatting and other tasks using LLM intent classification"""
        import re
        # Semantic intent classification using LLM
        intent_prompt = (
            "Classify the following user request by intent. Understand the Semantic Meaning behind the user's question. "
            "When the user asks for real-time data always give intent as web_search only."
            "Is it asking to edit or format a file, generate code, manage files, or something else? "
            "Return a JSON object with 'intent', 'filename' (if any), and 'reasoning'.\n"
            "You have Following Lits of intents (code_generation, file_management, file_editing ,research, data_analysis, automation, creative_writing, problem_solving, system_administration, web_scraping, web_search, api_integration, general)"
            f"Request: {user_request}"
        )
        try:
            class RequestIntent(BaseModel):
                intent:Literal["code_generation", "file_management", "file_editing" ,"research", "data_analysis", "automation", "creative_writing", "problem_solving", "system_administration", "web_scraping", "web_search", "api_integration", "general"]
                filename:str|None
                reasoning:str
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert intent classifier for agentic tasks."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intent_classification",
                        "schema": RequestIntent.model_json_schema()
                    }
                }
                
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            intent_result = json.loads(content)
            intent = intent_result.get("intent", "general")
            filename = intent_result.get("filename")
            reasoning = intent_result.get("reasoning", "")
            # Map intent to task_type and tools
            if intent in ["file_editing", "formatting"]:
                task_type = "file_editing"
                tools = ["filesystem", "edit_file", "llm_generation"]
                if not filename:
                    # Try to extract filename from user_request
                    file_match = re.search(r"\b([\w\-]+\.py)\b", user_request)
                    filename = file_match.group(1) if file_match else "unknown.py"
                reasoning = reasoning or f"LLM classified as file editing/formatting for {filename}"
            elif intent == "code_generation":
                task_type = "code_generation"
                tools = ["filesystem", "command"]
                reasoning = reasoning or "LLM classified as code generation"
            elif intent == "file_management":
                task_type = "file_management"
                tools = ["filesystem"]
                reasoning = reasoning or "LLM classified as file management"
            elif intent == "web_search":
                task_type = "web_search"
                tools = ["web"]
                reasoning = reasoning or "LLM classified as web_search (real-time data)"
            elif intent == "research":
                task_type = "research"
                tools = ["web", "data_processing"]
                reasoning = reasoning or "LLM classified as research"
            elif intent == "data_analysis":

                task_type = "data_analysis"
                tools = ["data_processing", "filesystem"]
                reasoning = reasoning or "LLM classified as data analysis"
            elif intent == "creative_writing":
                task_type = "creative_writing"
                tools = ["filesystem"]
                reasoning = reasoning or "LLM classified as creative writing"
            else:
                task_type = "general"
                tools = ["filesystem"]
                reasoning = reasoning or "LLM classified as general"
            return {
                "task_type": task_type,
                "complexity": "medium",
                "required_tools": tools,
                "subtasks": [user_request],
                "estimated_steps": 2,
                "reasoning": reasoning
            }
        except Exception as e:
            # Fallback to pattern-based analysis if LLM fails
            user_lower = user_request.lower()
            file_edit_keywords = ["edit", "format", "refactor", "clean up", "pep8", "lint", "beautify"]
            file_pattern = r"\b([\w\-]+\.py)\b"
            file_match = re.search(file_pattern, user_lower)
            if any(word in user_lower for word in file_edit_keywords) and file_match:
                filename = file_match.group(1)
                task_type = "file_editing"
                tools = ["filesystem", "edit_file", "llm_generation"]
                reasoning = f"Detected file editing/formatting task for {filename}"
            else:
                code_keywords = ["code", "program", "script", "develop", "python", "gui", "game", "create", "build", "generate"]
                realtime_keywords = [
                    "stock", "price", "current", "live", "today", "quote", "market", "weather", "news", "rate", "exchange", "crypto", "share", "value"
                ]
                if any(word in user_lower for word in code_keywords):
                    task_type = "code_generation"
                    tools = ["filesystem", "command"]
                    reasoning = "Detected as code_generation task based on keywords"
                elif any(word in user_lower for word in ["file", "directory", "folder", "organize"]):
                    task_type = "file_management"
                    tools = ["filesystem"]
                    reasoning = "Detected as file_management task based on keywords"
                elif any(word in user_lower for word in ["research", "find", "search", "information"]):
                    task_type = "research"
                    tools = ["web", "data_processing"]
                    reasoning = "Detected as research task based on keywords"
                elif any(word in user_lower for word in realtime_keywords):
                    task_type = "web_search"
                    tools = ["web"]
                    reasoning = "Detected as web_search (real-time data) based on keywords"
                elif any(word in user_lower for word in ["analyze", "data", "statistics", "process"]):
                    task_type = "data_analysis"
                    tools = ["data_processing", "filesystem"]
                    reasoning = "Detected as data_analysis task based on keywords"
                elif any(word in user_lower for word in ["write", "story", "article"]):
                    task_type = "creative_writing"
                    tools = ["filesystem"]
                    reasoning = "Detected as creative_writing task based on keywords"
                else:
                    task_type = "general"
                    tools = ["filesystem"]
                    reasoning = "General task analysis"
            return {
                "task_type": task_type,
                "complexity": "medium",
                "required_tools": tools,
                "subtasks": [user_request],
                "estimated_steps": 2,
                "reasoning": reasoning
            }

    
    def plan_execution(self, analysis: Dict[str, Any], user_request: str) -> List[Dict[str, Any]]:
        """Plan execution steps based on analysis"""
        
        system_prompt = """You are an execution planner. Create a detailed step-by-step plan to accomplish the task.

Return a JSON array of steps, each with:
- step: number
- action: tool name (filesystem,edit, command, web, data_processing, llm_generation)
- operation: specific operation
- parameters: parameters for the operation
- description: human-readable description
- depends_on: list of step numbers this depends on (optional)

Focus on practical, executable steps. Be specific with file paths and operations."""

        planning_prompt = f"""
User Request: {user_request}
Analysis: {json.dumps(analysis, indent=2)}

Use this Analysis and user request , understand the user intent and according to it.make a proper plan for this.

## Note :
- you have following actions to take tool name (filesystem, command, web, data_processing, llm_generation).
- take the proper actions as per user intent.
- when the user asks something about real-time data or factual data always try to do the web_seacrh for it.

Create an execution plan:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            plan = json.loads(content)
            
            # Validate plan structure
            if not isinstance(plan, list):
                return self._create_simple_plan(analysis, user_request)
            
            # Ensure each step has required fields
            for step in plan:
                step.setdefault("step", len(plan))
                step.setdefault("action", "filesystem")
                step.setdefault("operation", "create_file")
                step.setdefault("parameters", {})
                step.setdefault("description", "Execute step")
            
            return plan
            
        except Exception as e:
            return self._create_simple_plan(analysis, user_request)
    
    def _create_simple_plan(self, analysis: Dict[str, Any], user_request: str) -> List[Dict[str, Any]]:
        """Create a simple fallback plan"""
        task_type = analysis.get("task_type", "general")
        if task_type == "file_editing":
            import re
            user_text = analysis.get("subtasks", [""])[0]
            file_match = re.search(r"\b([\w\-]+\.py)\b", user_text)
            filename = file_match.group(1) if file_match else "unknown.py"
            return [
                {
                    "step": 1,
                    "action": "filesystem",
                    "operation": "read_file",
                    "parameters": {"path": filename},
                    "description": f"Read the content of {filename}"
                },
                {
                    "step": 2,
                    "action": "llm_generation",
                    "operation": "format_code",
                    "parameters": {
                        "request": "Format this Python code to PEP8 and clean up:\n{{step_1.data.content}}",
                        "language": "python"
                    },
                    "description": f"Format the code in {filename} using LLM"
                },
                {
                    "step": 3,
                    "action": "edit_file",
                    "operation": "edit_file",
                    "parameters": {
                        "file_path": filename,
                        "old_text": "{{step_1.data.content}}",
                        "new_text": "{{step_2.data.content}}",
                        "replace_all": True
                    },
                    "description": f"Write the formatted code back to {filename}"
                }
            ]
        elif task_type == "code_generation":
            return [
                {
                    "step": 1,
                    "action": "llm_generation",
                    "operation": "generate_code",
                    "parameters": {"request": user_request, "language": "python"},
                    "description": "Generate code based on request"
                },
                {
                    "step": 2,
                    "action": "filesystem",
                    "operation": "create_file",
                    "parameters": {"path": "generated_code.py", "content": "{{generated_content}}"},
                    "description": "Save generated code to file"
                }
            ]
        elif task_type == "file_management":
            return [
                {
                    "step": 1,
                    "action": "filesystem",
                    "operation": "list_directory",
                    "parameters": {"path": "."},
                    "description": "List current directory contents"
                }
            ]
        elif task_type == "web_search" or (task_type == "research" and "web" in analysis.get("required_tools", [])):
            return [
                {
                    "step": 1,
                    "action": "web",
                    "operation": "search",
                    "parameters": {"data": user_request, "description": user_request},
                    "description": "Search the web for real-time/factual data"
                }
            ]
        else:
            return [
                {
                    "step": 1,
                    "action": "llm_generation",
                    "operation": "generate_response",
                    "parameters": {"request": user_request},
                    "description": "Generate response to user request"
                }
            ]
    
    def generate_content(self, request: str, content_type: str = "general", **kwargs) -> LLMResponse:
        """Generate content based on request and type"""
        
        if content_type == "code":
            system_prompt = """You are an expert programmer. Generate clean, functional code.
- Include all necessary imports
- Add proper error handling
- Write clear comments
- Make code production-ready
- Return ONLY the code, no explanations"""
        elif content_type == "creative":
            system_prompt = """You are a creative writer. Generate engaging, original content.
- Be creative and imaginative
- Use vivid descriptions
- Maintain consistent tone
- Create compelling narratives"""
        else:
            system_prompt = """You are a helpful assistant. Provide accurate, useful responses.
- Be clear and concise
- Include relevant details
- Structure information logically
- Be practical and actionable"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 5000)
            )
            
            content = response.choices[0].message.content
            return LLMResponse(content=content, confidence=0.8)
            
        except Exception as e:
            return LLMResponse(
                content=f"Error generating content: {str(e)}",
                confidence=0.0
            )

class GeneralGroqAgent:
    """General-purpose agent powered by Groq LLM"""
    
    def __init__(self, groq_api_key: str):
        # Initialize tools
        self.tools = {
            'filesystem': FileSystemTool(),
            'edit_file': EditFileTool(),
            'command': CommandTool(),
            'web': WebTool(),
            'data_processing': DataProcessingTool()
        }
        
        # Initialize LLM engine
        self.llm = GroqLLMEngine(groq_api_key)
        
        # State management
        self.chat_history: List[Message] = []
        self.current_directory = os.getcwd()
        self.context = {"working_directory": self.current_directory}
        self.execution_history = []
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user request - main agent entry point (agentic loop)"""
        self.chat_history.append(Message("user", user_input))

        # Agentic loop
        MAX_ITERATIONS = 5
        state = {
            "user_input": user_input,
            "history": [],
            "goal_met": False,
            "final_response": None
        }
        for iteration in range(MAX_ITERATIONS):
            print(f"\n🧠 Agentic iteration {iteration+1}")
            analysis = self.llm.analyze_request(state["user_input"], list(self.tools.keys()))
            print(f"📋 Task type: {analysis.get('task_type', 'general')}")
            print("Analysis : ",analysis)
            print(f"📊 Complexity: {analysis.get('complexity', 'medium')}")
            plan = self.llm.plan_execution(analysis, state["user_input"])
            print(f"🔧 Planned {len(plan)} steps")
            print("After Planning State : ",state)
            results = self._execute_plan(plan, state["user_input"], analysis)
            state["history"].append({
                "iteration": iteration+1,
                "analysis": analysis,
                "plan": plan,
                "results": results
            })
            # Reflection: check if goal is met
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            if success_count == total_count and total_count > 0:
                state["goal_met"] = True
                state["final_response"] = self._generate_response(analysis, plan, results, state["user_input"])
                break
            # Optionally: ask for clarification if stuck (not implemented here, but can be added)
            # Re-plan: use intermediate results to inform next analysis
            # For now, just continue with the same user_input
        if not state["final_response"]:
            # If not successful after max iterations, summarize attempts
            last = state["history"][-1]
            state["final_response"] = self._generate_response(
                last["analysis"], last["plan"], last["results"], state["user_input"]
            )
        self.chat_history.append(Message("assistant", state["final_response"]))
        self.execution_history.append({
            "request": user_input,
            "agentic_history": state["history"],
            "goal_met": state["goal_met"],
            "timestamp": datetime.now().isoformat()
        })

        return state

    
    def _execute_plan(self, plan: List[Dict[str, Any]], user_request: str, analysis: Dict[str, Any]) -> List[ToolResult]:
        """Execute the planned steps with robust placeholder replacement and error handling"""
        import time
        results = []
        step_outputs = {}  # Track outputs for each step
        step_data = {}     # Track .data for each step
        generated_content = {}  # For backward compatibility

        def replace_placeholders(val: str) -> str:
            # Replace all {{step_X.output}} and {{step_X.data.key}} and {{generated_content}}
            def repl(match):
                placeholder = match.group(1)
                if placeholder == "generated_content":
                    return generated_content.get("generated_content", "[ERROR: No generated content]")
                # Match step_X.output or step_X.data.key
                m = re.match(r"step_(\d+)\.output", placeholder)
                if m:
                    step_num = int(m.group(1))
                    return step_outputs.get(step_num, f"[ERROR: step_{step_num}.output not found]")
                m = re.match(r"step_(\d+)\.data\.([\w_]+)", placeholder)
                if m:
                    step_num = int(m.group(1))
                    key = m.group(2)
                    return str(step_data.get(step_num, {}).get(key, f"[ERROR: step_{step_num}.data.{key} not found]"))
                return f"[ERROR: Unresolved placeholder {{{{{placeholder}}}}}]"  # Unresolved, show error
            # Replace all {{...}}
            return re.sub(r"{{\s*([^{}]+)\s*}}", repl, val)

        def llm_with_retry(request, content_type, max_retries=3):
            delay = 2
            response = None
            for attempt in range(max_retries):
                response = self.llm.generate_content(request, content_type)
                if response and not response.content.startswith("Error generating content"):
                    return response
                print(f"[WARN] LLM/API error: {response.content if response else 'No response'}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            # All retries failed
            return response if response else LLMResponse(content="Error generating content: No response", confidence=0.0)


        for idx, step in enumerate(plan):
            try:
                action = step.get("action")
                operation = step.get("operation")
                parameters = step.get("parameters", {})
                step_num = step.get("step", idx+1)

                print(f"  📌 Step {step_num}: {step.get('description', 'Executing...')}")

                # Replace placeholders in all string parameters
                for key, value in parameters.items():
                    if isinstance(value, str):
                        parameters[key] = replace_placeholders(value)

                # Handle content generation with retry
                if action=="web":
                    tool=self.tools[action]
                    result = tool.execute(**parameters)
                    step_outputs[step_num] = result.output
                    step_data[step_num] = result.data if result.data else {}
                    results.append(result)
                # Handle tool execution
                elif action == "llm_generation":
                    content_type = "code" if operation and "code" in operation else "general"
                    if "creative" in user_request.lower() or "write" in user_request.lower():
                        content_type = "creative"
                    llm_response = llm_with_retry(parameters.get("request", user_request), content_type)
                    generated_content["generated_content"] = llm_response.content
                    step_outputs[step_num] = llm_response.content
                    step_data[step_num] = {"content": llm_response.content}
                    # If LLM failed, mark as failed
                    if llm_response.content.startswith("Error generating content"):
                        results.append(ToolResult(
                            success=False,
                            output=llm_response.content,
                            error=llm_response.content,
                            data={"content": llm_response.content}
                        ))
                    else:
                        results.append(ToolResult(
                            success=True,
                            output=f"Content generated ({len(llm_response.content)} characters)",
                            data={"content": llm_response.content}
                        ))
                elif action in self.tools and action !="web" and action!="llm_generation":
                    tool = self.tools[action]
                    result = tool.execute(operation, **parameters)
                    # If writing to a file, ensure content is not a placeholder
                    if operation in ["create_file", "write_file"]:
                        content = parameters.get("content", "")
                        if "{{" in content:
                            # Unresolved placeholder, write error
                            result = tool.execute(operation, path=parameters.get("path", "output.txt"), content=replace_placeholders(content))
                    step_outputs[step_num] = result.output
                    step_data[step_num] = result.data if result.data else {}
                    results.append(result)
                else:
                    results.append(ToolResult(False, "", f"Unknown action: {action}"))
            except Exception as e:
                results.append(ToolResult(False, "", f"Step execution failed: {str(e)}"))
        return results


    
    def _generate_response(self, analysis: Dict[str, Any], plan: List[Dict[str, Any]], 
                          results: List[ToolResult], user_request: str) -> str:
        """Generate comprehensive response"""
        
        success_count = sum(1 for result in results if result.success)
        total_count = len(results)
        
        response_parts = []
        
        # Header based on task type
        task_type = analysis.get("task_type", "general")
        
        if success_count == total_count and total_count > 0:
            task_icons = {
                "code_generation": "💻",
                "file_management": "📁", 
                "research": "🔍",
                "data_analysis": "📊",
                "automation": "⚙️",
                "creative_writing": "✍️",
                "problem_solving": "🧩",
                "system_administration": "🛠️",
                "web_scraping": "🌐",
                "api_integration": "🔌"
            }
            icon = task_icons.get(task_type, "✅")
            response_parts.append(f"{icon} Task completed successfully!")
        elif success_count > 0:
            response_parts.append(f"⚡ Completed {success_count}/{total_count} steps")
        else:
            response_parts.append("❌ Task failed")
        
        # Show what was accomplished
        created_files = []
        processed_data = []
        executed_commands = []
        listed_items = []
        
        for i, (step, result) in enumerate(zip(plan, results)):
            if result.success:
                action = step.get("action")
                operation = step.get("operation")
                if action == "filesystem" and operation == "create_file":
                    file_path = step.get("parameters", {}).get("path")
                    if file_path:
                        created_files.append(file_path)
                elif action == "filesystem" and operation == "list_directory":
                    # Extract listed items from result.data
                    items = result.data.get("items", []) if result.data else []
                    listed_items.extend(items)
                elif action == "command":
                    cmd = step.get("parameters", {}).get("command", "")[:50]
                    executed_commands.append(cmd)
                elif action == "data_processing":
                    processed_data.append(step.get("description", "Data processed"))
        
        # Add specific accomplishments
        if created_files:
            response_parts.append(f"\n📁 Files created:")
            for file in created_files:
                response_parts.append(f"   • {file}")
        
        if listed_items:
            response_parts.append(f"\n📂 Directory listing:")
            for item in listed_items:
                name = item.get("name", "?")
                typ = item.get("type", "?")
                size = item.get("size", 0)
                response_parts.append(f"   • {name} [{typ}] ({size} bytes)")
        
        if executed_commands:
            response_parts.append(f"\n⚡ Commands executed:")
            for cmd in executed_commands[:3]:  # Limit display
                response_parts.append(f"   • {cmd}")
        
        if processed_data:
            response_parts.append(f"\n📊 Data processed:")
            for data in processed_data[:3]:
                response_parts.append(f"   • {data}")

        
        # Add execution summary
        response_parts.append(f"\n📋 Execution Summary:")
        for i, (step, result) in enumerate(zip(plan, results)):
            status = "✅" if result.success else "❌"
            desc = step.get("description", f"Step {i+1}")
            response_parts.append(f"  {status} {desc}")
            
            # Add error details for failed steps
            if not result.success and result.error:
                response_parts.append(f"      Error: {result.error[:100]}")
        
        # Add next steps or usage instructions
        if task_type == "code_generation" and created_files:
            response_parts.append(f"\n🚀 Next steps:")
            for file in created_files:
                if file.endswith('.py'):
                    response_parts.append(f"   python {file}")
                elif file.endswith('.js'):
                    response_parts.append(f"   node {file}")
                elif file.endswith('.html'):
                    response_parts.append(f"   Open {file} in browser")
        
        # Add insights from data analysis
        for result in results:
            if result.data and isinstance(result.data, dict):
                if "statistics" in str(result.data):
                    stats = result.data
                    response_parts.append(f"\n📈 Quick stats: {stats}")
                    break
        
        return '\n'.join(response_parts)
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get available tools and their descriptions"""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def get_execution_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear chat and execution history"""
        self.chat_history.clear()
        self.execution_history.clear()
    
    def add_tool(self, tool: BaseTool):
        """Add a new tool to the agent"""
        self.tools[tool.name] = tool
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent"""
        if tool_name in self.tools:
            del self.tools[tool_name]

# Enhanced CLI interface
class AgentCLI:
    """Command line interface for the general agent"""
    
    def __init__(self, agent: GeneralGroqAgent):
        self.agent = agent
        self.commands = {
            'help': self._show_help,
            'tools': self._show_tools,
            'history': self._show_history,
            'clear': self._clear_history,
            'status': self._show_status,
            'exit': self._exit,
            'quit': self._exit
        }
    
    def run(self):
        """Run the CLI interface"""
        self._show_welcome()
        
        while True:
            try:
                user_input = input("\n🎯 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in self.commands:
                    self.commands[user_input.lower()]()
                    continue
                
                # Process regular requests
                state = self.agent.process_request(user_input)
                final_response=""
                try:
                    content=state['history'][-1]['results'][-1].data['content']
                    final_flag=state['final_response']+"\n\n"
                    final_response=final_flag+content
                except:
                    final_response=state['final_response']
                print(f"\n🤖 Agent:\n{final_response}")
                print("\n" + "="*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Agent interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
    
    def _show_welcome(self):
        print("🤖 General Purpose AI Agent - Powered by Groq!")
        print("🌟 I can help with various tasks:")
        print("   💻 Code generation & development")
        print("   📁 File & directory management")
        print("   🔍 Research & web scraping")
        print("   📊 Data analysis & processing")
        print("   ✍️ Creative writing")
        print("   ⚙️ Task automation")
        print("   🛠️ System administration")
        print("\n💡 Examples:")
        print("   • 'Create a Python web scraper for news headlines'")
        print("   • 'Organize my downloads folder by file type'")
        print("   • 'Write a short story about space exploration'")
        print("   • 'Analyze this CSV file and show statistics'")
        print("   • 'Create a backup script for my documents'")
        print("   • 'Build a simple calculator GUI'")
        print("\nType 'help' for commands, 'quit' to exit")
        print("=" * 70)
    
    def _show_help(self):
        print("\n🔧 Available commands:")
        print("   help     - Show this help message")
        print("   tools    - List available tools")
        print("   history  - Show recent execution history")
        print("   clear    - Clear chat history")
        print("   status   - Show agent status")
        print("   quit/exit - Exit the agent")
        print("\n💬 Or just describe what you want me to do!")
    
    def _show_tools(self):
        print("\n🛠️ Available tools:")
        tools = self.agent.get_available_tools()
        for name, description in tools.items():
            print(f"   {name:15} - {description}")
    
    def _show_history(self):
        history = self.agent.get_execution_history()
        if not history:
            print("\n📝 No execution history yet")
            return
        
        print(f"\n📝 Recent execution history ({len(history)} items):")
        for i, entry in enumerate(history, 1):
            task_type = entry.get('analysis', {}).get('task_type', 'unknown')
            success_count = sum(1 for r in entry.get('results', []) if r.success)
            total_count = len(entry.get('results', []))
            
            print(f"   {i}. {task_type} - {success_count}/{total_count} steps successful")
            print(f"      Request: {entry.get('request', '')[:60]}...")
    
    def _clear_history(self):
        self.agent.clear_history()
        print("\n🗑️ History cleared")
    
    def _show_status(self):
        print(f"\n📊 Agent Status:")
        print(f"   Working directory: {self.agent.current_directory}")
        print(f"   Available tools: {len(self.agent.tools)}")
        print(f"   Chat history: {len(self.agent.chat_history)} messages")
        print(f"   Execution history: {len(self.agent.execution_history)} tasks")
    
    def _exit(self):
        print("\n👋 Thanks for using the General AI Agent!")
        exit(0)

# Main execution
def main():
    """Main function"""
    
    # Check for Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("   Get your API key from: https://console.groq.com/keys")
        print("   Then run: export GROQ_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize the agent
        agent = GeneralGroqAgent(groq_api_key)
        
        # Start CLI
        cli = AgentCLI(agent)
        cli.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("   Please check your API key and internet connection")

if __name__ == "__main__":
    main()