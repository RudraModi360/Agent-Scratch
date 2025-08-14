import os
import json
import re
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union, Callable, Literal, Tuple
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
    AUTOMATION = "automation"
    CREATIVE_WRITING = "creative_writing"
    PROBLEM_SOLVING = "problem_solving"
    SYSTEM_ADMINISTRATION = "system_administration"
    WEB_SCRAPING = "web_scraping"
    EDIT_FILE = "edit_file"
    API_INTEGRATION = "api_integration"
    WEB_SEARCH = "web_search"
    GENERAL = "general"
    COMPLEX_WORKFLOW = "complex_workflow"
    MULTI_DOMAIN = "multi_domain"

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"
    EXPERT = "expert"

class DecisionType(Enum):
    """Types of decisions the agent can make"""
    TOOL_SELECTION = "tool_selection"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    ERROR_RECOVERY = "error_recovery"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONTINUATION_DECISION = "continuation_decision"

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 1.0
    quality_score: float = 1.0

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
    """Enhanced context for task execution"""
    task_type: TaskType
    complexity: TaskComplexity
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    working_directory: str = "."
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

@dataclass
class Decision:
    """Represents a decision made by the agent"""
    decision_type: DecisionType
    decision: str
    reasoning: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ExecutionStep:
    """Enhanced execution step with decision tracking"""
    step: int
    action: str
    operation: str
    parameters: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    expected_outcome: str = ""
    success_criteria: List[str] = field(default_factory=list)
    fallback_actions: List[Dict[str, Any]] = field(default_factory=list)
    decision_points: List[Decision] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

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
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this tool provides"""
        return []
    
    def estimate_execution_time(self, **kwargs) -> float:
        """Estimate execution time in seconds"""
        return 1.0
    
    def assess_quality(self, result: ToolResult) -> float:
        """Assess the quality of the result (0-1)"""
        return 1.0 if result.success else 0.0

# Enhanced tools with better capabilities
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
            
            quality_score = 1.0 if replacement_count > 0 else 0.5
            
            if replacement_count > 0:
                return ToolResult(True, f"Replaced {replacement_count} occurrence(s) in {file_path}", 
                                quality_score=quality_score)
            else:
                return ToolResult(False, "", f"No occurrences of '{old_text}' found in {file_path}")
                
        except Exception as e:
            return ToolResult(False, "", f"Error: Failed to edit file - {e}")

    def get_capabilities(self) -> List[str]:
        return ["file_editing", "text_replacement", "code_modification"]

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
        """Execute file system operations with enhanced error handling"""
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
                result = operations[operation](**kwargs)
                # Assess quality based on operation success and data completeness
                result.quality_score = self.assess_quality(result)
                return result
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, "", str(e), confidence=0.0, quality_score=0.0)
    
    def get_capabilities(self) -> List[str]:
        return ["file_management", "directory_operations", "text_processing", "file_search"]
    
    def assess_quality(self, result: ToolResult) -> float:
        """Assess result quality based on completeness and accuracy"""
        if not result.success:
            return 0.0
        
        # Check if result has meaningful data
        if result.data:
            if isinstance(result.data, dict) and result.data:
                return 1.0
            elif isinstance(result.data, list) and result.data:
                return 1.0
        
        return 0.8 if result.output else 0.6

    # Implementation methods remain the same but with enhanced error handling
    def _create_file(self, path: str, content: str = '', **kwargs) -> ToolResult:
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            stats = os.stat(path)
            return ToolResult(True, f"File created: {path}", 
                            data={"path": path, "size": stats.st_size, "lines": len(content.splitlines())})
        except Exception as e:
            return ToolResult(False, "", str(e))

    def _write_file(self, path: str, content: str = '', **kwargs) -> ToolResult:
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            stats = os.stat(path)
            return ToolResult(True, f"File overwritten: {path}", 
                            data={"path": path, "size": stats.st_size, "lines": len(content.splitlines())})
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
                                "lines": len(content.splitlines()),
                                "words": len(content.split()),
                                "characters": len(content)
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
            
            return ToolResult(True, f"Found {len(files_info)} files matching pattern", 
                            data={"matches": files_info, "pattern": pattern, "total_matches": len(files_info)})
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
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        info["lines"] = len(content.splitlines())
                        info["words"] = len(content.split())
                        info["characters"] = len(content)
                        info["extension"] = os.path.splitext(path)[1]
                except:
                    info["readable"] = False
            
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
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "extension": os.path.splitext(item)[1] if os.path.isfile(full_path) else ""
                }
                items.append(item_info)
            
            return ToolResult(True, f"Listed {len(items)} items", 
                            data={
                                "items": items, 
                                "path": path,
                                "total_items": len(items),
                                "files": sum(1 for i in items if i["type"] == "file"),
                                "directories": sum(1 for i in items if i["type"] == "directory")
                            })
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
        return "Execute system commands, shell scripts, and manage processes with enhanced safety and monitoring"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_directory": {"type": "string", "default": "."},
                "timeout": {"type": "integer", "default": 30},
                "capture_output": {"type": "boolean", "default": True},
                "shell": {"type": "boolean", "default": True},
                "safe_mode": {"type": "boolean", "default": True}
            },
            "required": ["command"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            command = kwargs.get('command')
            safe_mode = kwargs.get('safe_mode', True)
            
            if not command:
                return ToolResult(False, '', 'Missing required parameter: command')
            
            # Enhanced safety checks
            if safe_mode and self._is_dangerous_command(command):
                return ToolResult(False, '', f'Command blocked for safety: {command}')
            
            start_time = datetime.now()
            result = subprocess.run(
                command,
                shell=kwargs.get('shell', True),
                capture_output=kwargs.get('capture_output', True),
                text=True,
                cwd=kwargs.get('working_directory', '.'),
                timeout=kwargs.get('timeout', 30)
            )
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            output = result.stdout
            error = result.stderr
            
            # Enhanced quality assessment
            quality_score = self._assess_command_quality(result, execution_time)
            
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                data={
                    "return_code": result.returncode,
                    "command": command,
                    "stdout": output,
                    "stderr": error,
                    "execution_time": execution_time,
                    "working_directory": kwargs.get('working_directory', '.')
                },
                confidence=0.9 if result.returncode == 0 else 0.3,
                quality_score=quality_score
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Command timed out", confidence=0.0, quality_score=0.0)
        except Exception as e:
            return ToolResult(False, "", str(e), confidence=0.0, quality_score=0.0)
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Check if command is potentially dangerous"""
        dangerous_patterns = [
            r'\brm\s+-rf\s+/',  # rm -rf /
            r'\bformat\b',      # format command
            r'\bdel\s+/[sqf]',  # Windows delete with flags
            r'\bsudo\s+rm',     # sudo rm
            r'>\s*/dev/',       # redirect to system devices
            r'\bdd\s+if=',      # disk dump
        ]
        
        return any(re.search(pattern, command, re.IGNORECASE) for pattern in dangerous_patterns)
    
    def _assess_command_quality(self, result: subprocess.CompletedProcess, execution_time: float) -> float:
        """Assess the quality of command execution"""
        if result.returncode != 0:
            return 0.0
        
        # Factor in execution time (faster is generally better for most commands)
        time_score = max(0.1, min(1.0, 10.0 / max(execution_time, 0.1)))
        
        # Factor in output usefulness
        output_score = 0.8 if result.stdout else 0.6
        
        return (time_score + output_score) / 2
    
    def get_capabilities(self) -> List[str]:
        return ["command_execution", "process_management", "system_interaction", "script_running"]

class WebTool(BaseTool):
    """Enhanced web operations tool for research and data gathering"""
    
    @property
    def name(self) -> str:
        return "web"
    
    @property
    def description(self) -> str:
        return "Perform web requests, download content, and intelligent web research with quality assessment"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "description": {"type": "string"},
                "search_depth": {"type": "string", "enum": ["quick", "thorough", "comprehensive"], "default": "thorough"}
            },
            "required": ["data", "description"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            data = kwargs.get("data", "")
            description = kwargs.get("description", "")
            search_depth = kwargs.get("search_depth", "thorough")

            if not data:
                return ToolResult(False, '', 'Missing data parameter for web search')

            # Build the enhanced user request string
            user_request = f"Search the web for: {data}. Additional context: {description}. Search depth: {search_depth}"

            # Get Groq API key from environment
            from os import getenv
            groq_api_key = getenv('GROQ_API_KEY')
            if not groq_api_key:
                return ToolResult(False, '', 'Missing GROQ_API_KEY environment variable')

            # Initialize Groq LLM Engine
            engine = GroqLLMEngine(groq_api_key)

            # Perform enhanced agentic tool call
            llm_response = engine.agentic_tool_call(engine.client, user_request)

            # Assess the quality of the response
            quality_score = self._assess_web_search_quality(llm_response, data)

            return ToolResult(
                success=True,
                output=llm_response.content,
                data={
                    "content": llm_response.content,
                    "reasoning": llm_response.reasoning if hasattr(llm_response, 'reasoning') else None,
                    "tool_calls": llm_response.tool_calls if hasattr(llm_response, 'tool_calls') else None,
                    "search_query": data,
                    "search_depth": search_depth,
                    "content_length": len(llm_response.content)
                },
                confidence=0.8,
                quality_score=quality_score
            )

        except Exception as e:
            return ToolResult(False, "", str(e), confidence=0.0, quality_score=0.0)
    
    def _assess_web_search_quality(self, response, query: str) -> float:
        """Assess the quality of web search results"""
        if not response or not hasattr(response, 'content'):
            return 0.0
        
        content = response.content
        if not content:
            return 0.0
        
        # Check content length (too short might be incomplete)
        length_score = min(1.0, len(content) / 500.0)  # Normalize around 500 chars
        
        # Check if the response seems relevant to the query
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        relevance_score = len(query_words.intersection(content_words)) / max(len(query_words), 1)
        
        return (length_score + relevance_score) / 2
    
    def get_capabilities(self) -> List[str]:
        return ["web_search", "content_retrieval", "research", "real_time_data"]

class DataProcessingTool(BaseTool):
    """Enhanced data analysis and processing tool"""
    
    @property
    def name(self) -> str:
        return "data_processing"
    
    @property
    def description(self) -> str:
        return "Process, analyze, and manipulate data (JSON, CSV, text) with advanced analytics and quality assessment"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string", 
                    "enum": ["parse_json", "parse_csv", "analyze_text", "transform_data", 
                           "statistics", "validate_data", "clean_data", "aggregate_data"]
                },
                "data": {"type": "string"},
                "format": {"type": "string", "enum": ["json", "csv", "text", "auto"]},
                "options": {"type": "object"},
                "validation_rules": {"type": "array"}
            },
            "required": ["operation", "data"]
        }
    
    def execute(self,operation:str,data:str, **kwargs) -> ToolResult:
        try:
            options = kwargs.get('options', {})
            
            if not operation:
                return ToolResult(False, '', 'Missing required parameter: operation')
            if data is None:
                return ToolResult(False, '', 'Missing required parameter: data')
            
            operations = {
                "parse_json": self._parse_json,
                "parse_csv": self._parse_csv,
                "analyze_text": self._analyze_text,
                "transform_data": self._transform_data,
                "statistics": self._calculate_statistics,
                "validate_data": self._validate_data,
                "clean_data": self._clean_data,
                "aggregate_data": self._aggregate_data
            }
            
            if operation in operations:
                result = operations[operation](data, options)
                result.quality_score = self._assess_processing_quality(result, operation)
                return result
            else:
                return ToolResult(False, "", f"Unknown data operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, "", str(e), confidence=0.0, quality_score=0.0)
    
    def _parse_json(self, data: str, options: Dict) -> ToolResult:
        try:
            parsed = json.loads(data)
            analysis = {
                "type": type(parsed).__name__,
                "size": len(str(parsed)),
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "length": len(parsed) if isinstance(parsed, (list, dict)) else None
            }
            return ToolResult(True, "JSON parsed successfully", 
                            data={"parsed": parsed, "analysis": analysis})
        except json.JSONDecodeError as e:
            return ToolResult(False, "", f"JSON parsing error: {str(e)}")
    
    def _parse_csv(self, data: str, options: Dict) -> ToolResult:
        try:
            import csv
            import io
            
            csv_file = io.StringIO(data)
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            
            analysis = {
                "total_rows": len(rows),
                "columns": reader.fieldnames,
                "column_count": len(reader.fieldnames) if reader.fieldnames else 0
            }
            
            return ToolResult(True, f"CSV parsed successfully - {len(rows)} rows", 
                            data={"parsed": rows, "analysis": analysis})
        except Exception as e:
            return ToolResult(False, "", f"CSV parsing error: {str(e)}")
    
    def _analyze_text(self, data: str, options: Dict) -> ToolResult:
        try:
            lines = data.splitlines()
            words = data.split()
            chars = len(data)
            
            # Advanced text analysis
            sentences = data.split('.')
            paragraphs = data.split('\n\n')
            
            analysis = {
                "lines": len(lines),
                "words": len(words),
                "characters": chars,
                "sentences": len([s for s in sentences if s.strip()]),
                "paragraphs": len([p for p in paragraphs if p.strip()]),
                "average_line_length": chars / max(len(lines), 1),
                "average_word_length": sum(len(word) for word in words) / max(len(words), 1),
                "most_common_words": self._get_word_frequency(words)[:10],
                "readability_score": self._calculate_readability(words, sentences)
            }
            
            return ToolResult(True, "Text analyzed successfully", data=analysis)
        except Exception as e:
            return ToolResult(False, "", f"Text analysis error: {str(e)}")
    
    def _transform_data(self, data: str, options: Dict) -> ToolResult:
        try:
            transform_type = options.get('transform_type', 'normalize')
            
            if transform_type == 'normalize':
                # Normalize text data
                normalized = data.lower().strip()
                return ToolResult(True, "Data normalized", data={"transformed": normalized})
            elif transform_type == 'extract_numbers':
                # Extract all numbers from text
                numbers = re.findall(r'-?\d+(?:\.\d+)?', data)
                return ToolResult(True, f"Extracted {len(numbers)} numbers", 
                                data={"numbers": [float(n) for n in numbers]})
            else:
                return ToolResult(False, "", f"Unknown transform type: {transform_type}")
                
        except Exception as e:
            return ToolResult(False, "", f"Data transformation error: {str(e)}")
    
    def _calculate_statistics(self, data: str, options: Dict) -> ToolResult:
        try:
            # Try to extract numbers from the data
            numbers = []
            
            # Try JSON first
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                    numbers = parsed
            except:
                pass
            
            # If not JSON, try to extract numbers from text
            if not numbers:
                numbers = [float(x) for x in data.split() if x.replace('.', '').replace('-', '').isdigit()]
            
            if not numbers:
                return ToolResult(False, "", "No numeric data found")
            
            import statistics
            stats = {
                "count": len(numbers),
                "sum": sum(numbers),
                "mean": statistics.mean(numbers),
                "median": statistics.median(numbers),
                "mode": statistics.mode(numbers) if len(set(numbers)) < len(numbers) else None,
                "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
                "variance": statistics.variance(numbers) if len(numbers) > 1 else 0,
                "min": min(numbers),
                "max": max(numbers),
                "range": max(numbers) - min(numbers)
            }
            
            return ToolResult(True, "Statistics calculated successfully", data=stats)
        except Exception as e:
            return ToolResult(False, "", f"Statistical analysis error: {str(e)}")
    
    def _validate_data(self, data: str, options: Dict) -> ToolResult:
        try:
            validation_rules = options.get('validation_rules', [])
            errors = []
            warnings = []
            
            # Basic validation
            if not data.strip():
                errors.append("Data is empty")
            
            # Custom validation rules
            for rule in validation_rules:
                rule_type = rule.get('type')
                if rule_type == 'min_length' and len(data) < rule.get('value', 0):
                    errors.append(f"Data length {len(data)} is below minimum {rule.get('value')}")
                elif rule_type == 'max_length' and len(data) > rule.get('value', 1000000):
                    warnings.append(f"Data length {len(data)} exceeds recommended maximum {rule.get('value')}")
            
            validation_result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "data_length": len(data),
                "data_type": type(data).__name__
            }
            
            return ToolResult(len(errors) == 0, "Data validation completed", data=validation_result)
        except Exception as e:
            return ToolResult(False, "", f"Data validation error: {str(e)}")
    
    def _clean_data(self, data: str, options: Dict) -> ToolResult:
        try:
            cleaned = data
            operations = []
            
            # Remove extra whitespace
            if options.get('remove_extra_whitespace', True):
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                operations.append("removed_extra_whitespace")
            
            # Remove special characters
            if options.get('remove_special_chars', False):
                cleaned = re.sub(r'[^\w\s]', '', cleaned)
                operations.append("removed_special_characters")
            
            # Convert to lowercase
            if options.get('lowercase', False):
                cleaned = cleaned.lower()
                operations.append("converted_to_lowercase")
            
            result_data = {
                "original_length": len(data),
                "cleaned_length": len(cleaned),
                "operations_performed": operations,
                "cleaned_data": cleaned
            }
            
            return ToolResult(True, f"Data cleaned - {len(operations)} operations performed", data=result_data)
        except Exception as e:
            return ToolResult(False, "", f"Data cleaning error: {str(e)}")
    
    def _aggregate_data(self, data: str, options: Dict) -> ToolResult:
        try:
            aggregation_type = options.get('type', 'summary')
            
            if aggregation_type == 'summary':
                # Provide a summary of the data
                char_count = len(data)
                word_count = len(data.split())
                line_count = len(data.splitlines())
                
                summary = {
                    "total_characters": char_count,
                    "total_words": word_count,
                    "total_lines": line_count,
                    "average_line_length": char_count / max(line_count, 1),
                    "data_preview": data[:200] + "..." if len(data) > 200 else data
                }
                
                return ToolResult(True, "Data aggregated into summary", data=summary)
            else:
                return ToolResult(False, "", f"Unknown aggregation type: {aggregation_type}")
                
        except Exception as e:
            return ToolResult(False, "", f"Data aggregation error: {str(e)}")
    
    def _get_word_frequency(self, words: List[str]) -> List[tuple]:
        """Calculate word frequency with enhanced filtering"""
        from collections import Counter
        
        # Clean words and filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        clean_words = []
        
        for word in words:
            cleaned = word.lower().strip('.,!?";:()[]{}')
            if len(cleaned) > 2 and cleaned not in stop_words:
                clean_words.append(cleaned)
        
        return Counter(clean_words).most_common()
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """Calculate a simple readability score"""
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score (lower is more readable)
        score = (avg_sentence_length * 0.4) + (avg_word_length * 0.6)
        return min(max(score, 1.0), 20.0)  # Clamp between 1 and 20
    
    def _assess_processing_quality(self, result: ToolResult, operation: str) -> float:
        """Assess the quality of data processing results"""
        if not result.success:
            return 0.0
        
        # Different quality metrics for different operations
        if operation in ['parse_json', 'parse_csv']:
            # For parsing, quality depends on successful parsing and data completeness
            if result.data and 'parsed' in result.data:
                return 1.0
            return 0.5
        
        elif operation == 'analyze_text':
            # For text analysis, quality depends on completeness of analysis
            if result.data and len(result.data) > 5:  # Multiple analysis metrics
                return 1.0
            return 0.7
        
        elif operation == 'statistics':
            # For statistics, quality depends on number of metrics calculated
            if result.data and len(result.data) > 5:
                return 1.0
            return 0.8
        
        return 0.8  # Default quality score
    
    def get_capabilities(self) -> List[str]:
        return ["data_parsing", "statistical_analysis", "text_analysis", "data_validation", 
                "data_cleaning", "data_transformation"]

class EnhancedGroqLLMEngine:
    """Enhanced Groq LLM integration with advanced decision making"""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.decision_history: List[Decision] = []
        self.context_memory = {}
    
    def analyze_request(self, user_request: str, available_tools: Optional[List[str]] = None, 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced request analysis with deep intent understanding"""
        
        system_prompt = """You are an advanced task analysis expert with deep understanding of user intent and multi-step planning capabilities.

Your task is to analyze user requests and provide comprehensive analysis including:
1. Primary and secondary task types
2. Complexity assessment (simple/medium/complex/multi_step/expert)
3. Required tools and their priorities
4. Detailed subtasks breakdown
5. Dependencies and constraints
6. Success criteria and quality metrics
7. Risk assessment and mitigation strategies
8. Alternative approaches

Available task types: code_generation, file_management, edit_file, research, data_analysis, automation, 
creative_writing, problem_solving, system_administration, web_scraping, web_search, api_integration, 
general, complex_workflow, multi_domain

Available tools: """ + str(available_tools or [])

        # Build context-aware prompt
        context_info = ""
        if context:
            context_info = f"\nContext from previous interactions: {json.dumps(context, indent=2)}"

        user_prompt = f"""
        Analyze this request with deep intent understanding: {user_request}
        
        Available tools: {available_tools}
        {context_info}
        
        Provide a comprehensive JSON analysis with:
        - primary_task_type: Main task category
        - secondary_task_types: Additional relevant categories (array)
        - complexity: Detailed complexity assessment
        - required_tools: Prioritized list with usage rationale
        - subtasks: Detailed breakdown with dependencies
        - estimated_steps: Realistic step count with reasoning
        - success_criteria: Measurable success indicators
        - quality_metrics: How to assess output quality
        - risks: Potential issues and mitigation
        - alternatives: Alternative approaches
        - reasoning: Detailed analysis explanation
        - confidence: Your confidence in this analysis (0-1)
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            analysis = self._parse_analysis_response(content)
            
            # Store analysis in context memory for future reference
            self.context_memory['last_analysis'] = analysis
            
            return analysis
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._enhanced_fallback_analysis(user_request, available_tools, context)
    
    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM analysis response with error handling"""
        try:
            # Clean JSON markers
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            analysis = json.loads(content)
            
            # Ensure all required fields exist with defaults
            required_fields = {
                "primary_task_type": "general",
                "secondary_task_types": [],
                "complexity": "medium", 
                "required_tools": ["filesystem"],
                "subtasks": [],
                "estimated_steps": 1,
                "success_criteria": ["Task completed"],
                "quality_metrics": ["Output generated"],
                "risks": ["Standard execution risks"],
                "alternatives": ["Direct approach"],
                "reasoning": "General task analysis",
                "confidence": 0.7
            }
            
            for field, default in required_fields.items():
                analysis.setdefault(field, default)
            
            # Validate complexity
            valid_complexity = ["simple", "medium", "complex", "multi_step", "expert"]
            if analysis["complexity"] not in valid_complexity:
                analysis["complexity"] = "medium"
            
            return analysis
            
        except json.JSONDecodeError:
            return self._enhanced_fallback_analysis("", [], {})
    
    def _enhanced_fallback_analysis(self, user_request: str, available_tools: Optional[List[str]] = None,
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced fallback analysis using semantic understanding"""
        
        # Use LLM for intent classification even if JSON parsing failed
        intent_prompt = f"""
        Classify the user request intent and provide structured analysis:
        Request: {user_request}
        
        Respond with intent classification considering:
        - File operations (create, edit, read, organize)
        - Code development (write, debug, refactor)
        - Research tasks (search, analyze, summarize)
        - Data processing (parse, analyze, visualize)
        - Creative tasks (write, generate, design)
        - System tasks (automate, configure, manage)
        - Multi-step workflows
        
        Consider complexity indicators:
        - Multiple steps mentioned
        - Integration requirements
        - Error handling needs
        - Quality requirements
        - Time constraints
        """
        
        try:
            class EnhancedRequestIntent(BaseModel):
                primary_intent: Literal["code_generation", "file_management", "file_editing", "research", 
                                      "data_analysis", "automation", "creative_writing", "problem_solving", 
                                      "system_administration", "web_scraping", "web_search", "api_integration", 
                                      "general", "complex_workflow", "multi_domain"]
                complexity_level: Literal["simple", "medium", "complex", "multi_step", "expert"]
                key_requirements: List[str]
                estimated_difficulty: float  # 0-1 scale
                reasoning: str
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert intent classifier for complex task analysis."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.0,
                max_tokens=500,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "enhanced_intent_classification",
                        "schema": EnhancedRequestIntent.model_json_schema()
                    }
                }
            )
            
            content = response.choices[0].message.content.strip()
            intent_result = json.loads(content)
            
            # Map to our analysis format
            primary_task = intent_result.get("primary_intent", "general")
            complexity = intent_result.get("complexity_level", "medium")
            requirements = intent_result.get("key_requirements", [])
            
            # Determine tools based on intent
            tool_mapping = {
                "file_editing": ["filesystem", "edit_file"],
                "code_generation": ["filesystem", "command"],
                "file_management": ["filesystem"],
                "web_search": ["web"],
                "research": ["web", "data_processing"],
                "data_analysis": ["data_processing", "filesystem"],
                "creative_writing": ["filesystem"],
                "complex_workflow": ["filesystem", "command", "web", "data_processing"],
                "multi_domain": ["filesystem", "command", "web", "data_processing", "edit_file"]
            }
            
            required_tools = tool_mapping.get(primary_task, ["filesystem"])
            
            return {
                "primary_task_type": primary_task,
                "secondary_task_types": [],
                "complexity": complexity,
                "required_tools": required_tools,
                "subtasks": requirements,
                "estimated_steps": len(requirements) if requirements else 2,
                "success_criteria": [f"Complete {primary_task} task"],
                "quality_metrics": ["Output accuracy", "Task completion"],
                "risks": ["Execution errors", "Incomplete results"],
                "alternatives": ["Alternative tool usage", "Manual approach"],
                "reasoning": intent_result.get("reasoning", "Fallback analysis based on intent classification"),
                "confidence": max(0.6, intent_result.get("estimated_difficulty", 0.7))
            }
            
        except Exception as e:
            # Ultimate fallback to pattern-based analysis
            return self._pattern_based_analysis(user_request)
    
    def _pattern_based_analysis(self, user_request: str) -> Dict[str, Any]:
        """Pattern-based analysis as ultimate fallback"""
        user_lower = user_request.lower()
        
        # Multi-step indicators
        multi_step_indicators = ["then", "after", "next", "following", "subsequently", "pipeline", "workflow"]
        is_multi_step = any(indicator in user_lower for indicator in multi_step_indicators)
        
        # Complexity indicators
        complex_indicators = ["integrate", "optimize", "advanced", "sophisticated", "comprehensive", "robust"]
        is_complex = any(indicator in user_lower for indicator in complex_indicators)
        
        # Determine complexity
        if is_multi_step and is_complex:
            complexity = "expert"
        elif is_multi_step:
            complexity = "multi_step"
        elif is_complex:
            complexity = "complex"
        else:
            complexity = "medium"
        
        # Task type determination
        task_patterns = {
            "code_generation": ["create", "build", "develop", "program", "code", "script", "application"],
            "file_editing": ["edit", "modify", "update", "change", "fix", "format"],
            "research": ["research", "find", "search", "investigate", "study"],
            "data_analysis": ["analyze", "process", "statistics", "data", "calculate"],
            "web_search": ["current", "latest", "real-time", "today", "now", "recent"],
            "automation": ["automate", "schedule", "batch", "bulk", "repeat"],
        }
        
        primary_task = "general"
        for task_type, patterns in task_patterns.items():
            if any(pattern in user_lower for pattern in patterns):
                primary_task = task_type
                break
        
        return {
            "primary_task_type": primary_task,
            "secondary_task_types": [],
            "complexity": complexity,
            "required_tools": ["filesystem", "command"] if primary_task == "code_generation" else ["filesystem"],
            "subtasks": [user_request],
            "estimated_steps": 3 if is_multi_step else 2,
            "success_criteria": ["Task completed successfully"],
            "quality_metrics": ["Output generated", "Requirements met"],
            "risks": ["Execution failure", "Incomplete output"],
            "alternatives": ["Manual approach", "Alternative tools"],
            "reasoning": f"Pattern-based analysis identified {primary_task} with {complexity} complexity",
            "confidence": 0.6
        }
    
    def make_decision(self, decision_type: DecisionType, context: Dict[str, Any], 
                     options: List[str]) -> Decision:
        """Make intelligent decisions based on context and options"""
        
        decision_prompt = f"""
        You need to make a {decision_type.value} decision.
        
        Context: {json.dumps(context, indent=2)}
        Available options: {options}
        
        Consider:
        1. Current situation and constraints
        2. Success probability of each option
        3. Risk assessment
        4. Resource requirements
        5. Long-term impact
        
        Provide your decision with detailed reasoning.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert decision maker for AI agent operations."},
                    {"role": "user", "content": decision_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract decision and reasoning
            decision_text = options[0]  # Default to first option
            reasoning = content
            confidence = 0.7
            
            # Try to extract specific decision if mentioned
            for option in options:
                if option.lower() in content.lower():
                    decision_text = option
                    confidence = 0.8
                    break
            
            decision = Decision(
                decision_type=decision_type,
                decision=decision_text,
                reasoning=reasoning,
                confidence=confidence,
                alternatives=[opt for opt in options if opt != decision_text]
            )
            
            self.decision_history.append(decision)
            return decision
            
        except Exception as e:
            # Fallback decision
            return Decision(
                decision_type=decision_type,
                decision=options[0] if options else "default",
                reasoning=f"Fallback decision due to error: {str(e)}",
                confidence=0.5,
                alternatives=options[1:] if len(options) > 1 else []
            )
    
    def plan_execution(self, analysis: Dict[str, Any], user_request: str, 
                      context: Optional[Dict[str, Any]] = None) -> List[ExecutionStep]:
        """Enhanced execution planning with decision points and fallbacks"""
        
        complexity = analysis.get("complexity", "medium")
        primary_task = analysis.get("primary_task_type", "general")
        
        system_prompt = f"""You are an advanced execution planner for AI agents. Create detailed execution plans with: 1. Step-by-step breakdown 2. Decision points and alternatives 3. Error handling and fallbacks 4. Success criteria for each step 5. Dependencies and prerequisites 6. Quality checkpoints Consider the task complexity ({complexity}) and type ({primary_task}). You have following tools and they have following options available :
TOOL_ACTIONS =[
    "filesystem":[
        "actions": [
            "create_file",      # Create a new file
            "read_file",        # Read contents of a file
            "write_file",       # Overwrite file contents
            "append_file",      # Append to file
            "delete_file",      # Delete file or directory
            "create_directory", # Create a new directory
            "list_directory",   # List contents of a directory
            "search_files",     # Search for files by pattern
            "file_stats",       # Get file statistics
            "copy_file",        # Copy file
            "move_file"         # Move file
        ],
        "cli_examples": [
            "touch filename.txt",           # create_file
            "cat filename.txt",             # read_file
            "echo 'text' > filename.txt",   # write_file
            "echo 'text' >> filename.txt",  # append_file
            "rm filename.txt",              # delete_file
            "mkdir new_folder",             # create_directory
            "ls new_folder",                # list_directory
            "find . -name '*.py'",          # search_files
            "stat filename.txt",            # file_stats
            "cp src.txt dest.txt",          # copy_file
            "mv src.txt dest.txt"           # move_file
        ]
    ],
    "edit_file":[
        "actions": [
            "edit_file"  # Modify existing file by exact text replacement
        ],
        "cli_examples": [
            "sed -i 's/old_text/new_text/' filename.txt",  # edit_file (single occurrence)
            "sed -i 's/old_text/new_text/g' filename.txt"  # edit_file (all occurrences)
        ]
    ],
    "command":[
        "actions": [
            "run_command",      # Execute a shell/system command
            "run_script",       # Run a shell script
            "manage_process"    # Manage system processes
        ],
        "cli_examples": [
            "ls",                       # run_command (list files)
            "pwd",                      # run_command (print working directory)
            "echo Hello World",         # run_command (print text)
            "python script.py",         # run_script (run Python script)
            "bash script.sh",           # run_script (run Bash script)
            "ps aux",                   # manage_process (list processes)
            "kill <pid>",               # manage_process (kill process)
            "timeout 10s command",      # run_command with timeout
        ]
    ],
    "web":[
        "actions": [
            "web_search",       # Search the web for information
            "download_content", # Download web content
            "fetch_data"        # Fetch data from a URL
        ],
        "cli_examples": [
            "curl https://example.com",         # fetch_data/download_content
            "wget https://example.com/file.zip" # download_content
            # For web_search, typically done via browser or search engine API
        ]
    ],
    "data_processing":[
        "actions": [
            "parse_json",       # Parse JSON data
            "parse_csv",        # Parse CSV data
            "analyze_text",     # Analyze text (semantic, statistical, etc.)
            "transform_data",   # Transform data (normalize, extract, etc.)
            "statistics",       # Calculate statistics
            "validate_data",    # Validate data against rules
            "clean_data",       # Clean/normalize data
            "aggregate_data"    # Aggregate data
        ],
        "cli_examples": [
            "jq . file.json",                   # parse_json
            "csvtool col 1,2 file.csv",         # parse_csv
            "wc -w file.txt",                   # analyze_text (word count)
            # validate_data, clean_data, aggregate_data are usually done via scripts or specialized tools
        ]
    ]
]

Return a JSON array of steps with enhanced structure:
- step: number
- action: tool name
- operation: specific operation
- parameters: operation parameters
- description: human-readable description
- depends_on: prerequisite steps (optional)
- expected_outcome: what should happen
- success_criteria: measurable success indicators
- fallback_actions: alternative approaches if main action fails
- decision_points: points where agent should make decisions
- max_retries: maximum retry attempts

Focus on robust, production-ready execution with comprehensive error handling."""

        available_tools=['filesystem', 'command', 'web', 'data_processing', 'edit_file']

        planning_prompt = f"""
        
        User Request:{user_request} break it down into easily executable steps using available tools : {available_tools}
        Analysis: {json.dumps(analysis, indent=2)}
        Context: {json.dumps(context or {}, indent=2)}
        
        Create a detailed execution plan considering:
        - Multi-step workflows for complex tasks
        - Quality validation at each step
        - Error recovery mechanisms
        - Alternative approaches
        - Resource optimization
        
        Available actions: filesystem, command, web, data_processing, edit_file, llm_generation
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            plan_data = json.loads(content)
            
            # Convert to ExecutionStep objects
            execution_steps = []
            for i, step_data in enumerate(plan_data):
                step = ExecutionStep(
                    step=step_data.get("step", i + 1),
                    action=step_data.get("action", "filesystem"),
                    operation=step_data.get("operation", "create_file"),
                    parameters=step_data.get("parameters", {}),
                    description=step_data.get("description", f"Execute step {i + 1}"),
                    depends_on=step_data.get("depends_on", []),
                    expected_outcome=step_data.get("expected_outcome", "Step completed"),
                    success_criteria=step_data.get("success_criteria", ["Output generated"]),
                    fallback_actions=step_data.get("fallback_actions", []),
                    max_retries=step_data.get("max_retries", 3)
                )
                execution_steps.append(step)
            
            return execution_steps
            
        except Exception as e:
            return self._create_enhanced_simple_plan(analysis, user_request)
    
    def _create_enhanced_simple_plan(self, analysis: Dict[str, Any], user_request: str) -> List[ExecutionStep]:
        """Create enhanced simple plan with better decision making"""
        task_type = analysis.get("primary_task_type", "general")
        complexity = analysis.get("complexity", "medium")
        
        if task_type == "file_editing":
            filename = self._extract_filename(user_request)
            return [
                ExecutionStep(
                    step=1,
                    action="filesystem",
                    operation="read_file",
                    parameters={"path": filename},
                    description=f"Read current content of {filename}",
                    expected_outcome="File content retrieved",
                    success_criteria=["File exists and readable", "Content loaded successfully"],
                    fallback_actions=[{"action": "filesystem", "operation": "create_file", "parameters": {"path": filename, "content": ""}}]
                ),
                ExecutionStep(
                    step=2,
                    action="llm_generation",
                    operation="format_code",
                    parameters={
                        "request": f"Improve and format this code according to best practices:\n{{{{step_1.data.content}}}}",
                        "language": "python"
                    },
                    description="Format and improve the code",
                    depends_on=[1],
                    expected_outcome="Well-formatted, improved code",
                    success_criteria=["Code is syntactically valid", "Follows style guidelines", "Improvements made"],
                    fallback_actions=[{"action": "llm_generation", "operation": "basic_format", "parameters": {"request": "Basic formatting only"}}]
                ),
                ExecutionStep(
                    step=3,
                    action="edit_file",
                    operation="edit_file",
                    parameters={
                        "file_path": filename,
                        "old_text": "{{step_1.data.content}}",
                        "new_text": "{{step_2.data.content}}"
                    },
                    description=f"Write improved code back to {filename}",
                    depends_on=[1, 2],
                    expected_outcome="File updated with improved code",
                    success_criteria=["File successfully updated", "No data loss", "Improvements preserved"],
                    fallback_actions=[{"action": "filesystem", "operation": "create_file", "parameters": {"path": f"{filename}.backup", "content": "{{step_1.data.content}}"}}]
                )
            ]
        
        # Add more enhanced plans for other task types...
        return [
            ExecutionStep(
                step=1,
                action="llm_generation",
                operation="generate_response",
                parameters={"request": user_request},
                description="Generate response to user request",
                expected_outcome="Relevant response generated",
                success_criteria=["Response addresses user request", "Content is coherent"],
                fallback_actions=[{"action": "llm_generation", "operation": "simple_response", "parameters": {"request": "Provide basic response"}}]
            )
        ]
    
    def _extract_filename(self, text: str) -> str:
        """Extract filename from user request"""
        file_match = re.search(r"\b([\w\-]+\.py)\b", text)
        return file_match.group(1) if file_match else "code.py"
    
    def agentic_tool_call(self, client: Groq, user_input: str):
        """Enhanced agentic tool call with better error handling"""
        try:
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
        except Exception as e:
            # Fallback to regular model if compound-beta fails
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that provides accurate and comprehensive responses to user queries."
                        },
                        {
                            "role": "user", 
                            "content": user_input
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message
            except Exception as fallback_error:
                # Return error message as mock response
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                        self.reasoning = None
                        self.tool_calls = None
                
                return MockResponse(f"Error performing web search: {str(fallback_error)}")
    
    def generate_content(self, request: str, content_type: str = "general", **kwargs) -> LLMResponse:
        """Enhanced content generation with retry logic and quality assessment"""
        
        system_prompts = {
            "code": """You are an expert programmer and software architect. Generate clean, efficient, well-documented code.
            
Requirements:
- Include comprehensive error handling
- Add detailed comments and docstrings
- Follow best practices and design patterns
- Make code production-ready and maintainable
- Include necessary imports and dependencies
- Provide usage examples when appropriate
- Consider performance and security implications
            
Return ONLY the code with comments, no additional explanations.""",

            "creative": """You are a masterful creative writer with expertise in storytelling, prose, and imaginative content.
            
Guidelines:
- Create engaging, original content with vivid imagery
- Develop compelling characters and narratives
- Use rich, descriptive language
- Maintain consistent tone and style
- Create emotional resonance with readers
- Show don't tell where appropriate
- Craft memorable and impactful content""",

            "analysis": """You are a data analysis expert providing comprehensive insights and interpretations.
            
Approach:
- Provide thorough analysis with clear methodology
- Include statistical insights where relevant
- Identify patterns, trends, and anomalies
- Offer actionable recommendations
- Present findings in structured, accessible format
- Consider multiple perspectives and implications
- Validate conclusions with evidence""",

            "general": """You are a knowledgeable AI assistant providing accurate, helpful responses.
            
Guidelines:
- Be clear, concise, and well-structured
- Provide practical, actionable information
- Include relevant details and context
- Consider different viewpoints when appropriate
- Ensure accuracy and reliability
- Tailor response to user's expertise level"""
        }
        
        system_prompt = system_prompts.get(content_type, system_prompts["general"])
        
        # Enhanced request with context
        enhanced_request = request
        if kwargs.get('context'):
            enhanced_request = f"Context: {kwargs['context']}\n\nRequest: {request}"
        
        max_retries = kwargs.get('max_retries', 3)
        temperature = kwargs.get('temperature', 0.7)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": enhanced_request}
                    ],
                    temperature=temperature,
                    max_tokens=kwargs.get('max_tokens', 5000)
                )
                
                content = response.choices[0].message.content
                
                # Assess content quality
                quality_score = self._assess_content_quality(content, content_type, request)
                confidence = min(0.9, 0.6 + (quality_score * 0.3))
                
                # If quality is too low and we have retries left, try again with adjusted parameters
                if quality_score < 0.5 and attempt < max_retries - 1:
                    temperature = max(0.3, temperature - 0.2)  # Reduce randomness
                    continue
                
                return LLMResponse(
                    content=content, 
                    confidence=confidence,
                    reasoning=f"Content generated with quality score: {quality_score:.2f}"
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return LLMResponse(
                        content=f"Error generating content after {max_retries} attempts: {str(e)}",
                        confidence=0.0,
                        reasoning=f"Failed after {max_retries} retry attempts"
                    )
                
                # Wait before retry
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return LLMResponse(
            content="Failed to generate content",
            confidence=0.0,
            reasoning="All retry attempts exhausted"
        )
    
    def _assess_content_quality(self, content: str, content_type: str, original_request: str) -> float:
        """Assess the quality of generated content"""
        if not content or len(content.strip()) < 10:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length assessment (context-dependent)
        if content_type == "code":
            # Code should be substantial but not unnecessarily verbose
            if 50 <= len(content) <= 5000:
                quality_score += 0.2
            elif len(content) > 20:
                quality_score += 0.1
        else:
            # Other content types
            if 100 <= len(content) <= 3000:
                quality_score += 0.2
            elif len(content) > 50:
                quality_score += 0.1
        
        # Relevance check (simple keyword matching)
        request_words = set(original_request.lower().split())
        content_words = set(content.lower().split())
        relevance = len(request_words.intersection(content_words)) / max(len(request_words), 1)
        quality_score += min(0.3, relevance * 0.5)
        
        # Structure assessment
        if content_type == "code":
            # Check for code structure indicators
            code_indicators = ['def ', 'class ', 'import ', 'if ', 'for ', 'while ']
            if any(indicator in content for indicator in code_indicators):
                quality_score += 0.1
        
        return min(1.0, quality_score)

# Create alias for backward compatibility
GroqLLMEngine = EnhancedGroqLLMEngine

class EnhancedGeneralGroqAgent:
    """Enhanced general-purpose agent with advanced decision making capabilities"""
    
    def __init__(self, groq_api_key: str):
        # Initialize enhanced tools
        self.tools = {
            'filesystem': FileSystemTool(),
            'edit_file': EditFileTool(),
            'command': CommandTool(),
            'web': WebTool(),
            'data_processing': DataProcessingTool()
        }
        
        # Initialize enhanced LLM engine
        self.llm = EnhancedGroqLLMEngine(groq_api_key)
        
        # Enhanced state management
        self.chat_history: List[Message] = []
        self.current_directory = os.getcwd()
        self.context = {"working_directory": self.current_directory}
        self.execution_history = []
        self.decision_history: List[Decision] = []
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "average_execution_time": 0,
            "quality_scores": []
        }
        
        # Learning and adaptation
        self.user_preferences = {}
        self.common_patterns = {}
        self.error_patterns = {}
    
    def process_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced request processing with multi-step decision making"""
        start_time = datetime.now()
        self.chat_history.append(Message("user", user_input))
        self.performance_metrics["total_tasks"] += 1

        # Enhanced agentic loop with decision tracking
        MAX_ITERATIONS = 7  # Increased for complex tasks
        state = {
            "user_input": user_input,
            "context": context or {},
            "history": [],
            "goal_met": False,
            "final_response": None,
            "decisions_made": [],
            "quality_assessment": {},
            "adaptation_notes": []
        }
        
        for iteration in range(MAX_ITERATIONS):
            print(f"\n🧠 Enhanced agentic iteration {iteration+1}/{MAX_ITERATIONS}")
            
            # Enhanced analysis with context
            analysis = self.llm.analyze_request(
                state["user_input"], 
                list(self.tools.keys()),
                {**state["context"], **self.context}
            )
            
            print(f"📋 Primary task: {analysis.get('primary_task_type', 'general')}")
            print(f"📊 Complexity: {analysis.get('complexity', 'medium')}")
            print(f"🎯 Confidence: {analysis.get('confidence', 0.7):.2f}")
            
            # Decision point: Should we continue with this analysis?
            if analysis.get('confidence', 0.7) < 0.4:
                decision = self.llm.make_decision(
                    DecisionType.STRATEGY_ADAPTATION,
                    {"low_confidence_analysis": analysis, "iteration": iteration},
                    ["continue_with_analysis", "request_clarification", "use_fallback_approach"]
                )
                state["decisions_made"].append(decision)
                
                if decision.decision == "request_clarification":
                    state["final_response"] = "I need more clarification about your request. Could you provide more specific details?"
                    break
            
            # Enhanced planning
            plan = self.llm.plan_execution(analysis, state["user_input"], state["context"])
            print(f"🔧 Planned {len(plan)} enhanced steps")
            print("📋 Designed Plan: ", plan)
            
            # Execute plan with enhanced monitoring
            results = self._execute_enhanced_plan(plan, state["user_input"], analysis)
            print(f"📈 Results of iteration {iteration+1}: ", results[-1].data['content'])
            # Quality assessment
            quality_assessment = self._assess_execution_quality(results, analysis)
            state["quality_assessment"] = quality_assessment
            
            # Store iteration data
            state["history"].append({
                "iteration": iteration+1,
                "analysis": analysis,
                "plan": plan,
                "results": results,
                "quality_assessment": quality_assessment,
                "decisions": [d for d in self.llm.decision_history if d not in state["decisions_made"]]
            })
            
            # Enhanced success evaluation
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            avg_quality = sum(r.quality_score for r in results) / max(total_count, 1)
            
            # Decision point: Is this execution satisfactory?
            if success_count == total_count and total_count > 0:
                quality_decision = self.llm.make_decision(
                    DecisionType.QUALITY_ASSESSMENT,
                    {
                        "success_rate": success_count / total_count,
                        "average_quality": avg_quality,
                        "iteration": iteration
                    },
                    ["accept_results", "attempt_improvement", "require_higher_quality"]
                )
                state["decisions_made"].append(quality_decision)
                
                if quality_decision.decision == "accept_results" or avg_quality > 0.7:
                    state["goal_met"] = True
                    state["final_response"] = self._generate_enhanced_response(
                        analysis, plan, results, state["user_input"], state
                    )
                    break
                elif quality_decision.decision == "attempt_improvement" and iteration < MAX_ITERATIONS - 2:
                    # Modify approach for next iteration
                    state["user_input"] = f"Improve previous attempt: {state['user_input']}"
                    continue
            
            # Decision point: Should we continue or adapt strategy?
            if iteration >= MAX_ITERATIONS - 2:
                continuation_decision = self.llm.make_decision(
                    DecisionType.CONTINUATION_DECISION,
                    {
                        "iterations_completed": iteration + 1,
                        "success_rate": success_count / max(total_count, 1),
                        "quality_score": avg_quality
                    },
                    ["finalize_current_results", "one_more_attempt", "declare_partial_success"]
                )
                state["decisions_made"].append(continuation_decision)
                
                if continuation_decision.decision != "one_more_attempt":
                    break
        
        # Finalize response if not already set
        if not state["final_response"]:
            last = state["history"][-1] if state["history"] else {}
            state["final_response"] = self._generate_enhanced_response(
                last.get("analysis", {}), 
                last.get("plan", []), 
                last.get("results", []), 
                state["user_input"], 
                state
            )
        
        # Update performance metrics
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if state["goal_met"]:
            self.performance_metrics["successful_tasks"] += 1
        
        self.performance_metrics["average_execution_time"] = (
            (self.performance_metrics["average_execution_time"] * (self.performance_metrics["total_tasks"] - 1) + execution_time) /
            self.performance_metrics["total_tasks"]
        )
        
        if state["quality_assessment"]:
            self.performance_metrics["quality_scores"].append(state["quality_assessment"].get("overall_quality", 0.5))
        
        # Store execution history with enhanced data
        self.chat_history.append(Message("assistant", state["final_response"]))
        self.execution_history.append({
            "request": user_input,
            "agentic_history": state["history"],
            "goal_met": state["goal_met"],
            "decisions_made": state["decisions_made"],
            "quality_assessment": state["quality_assessment"],
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Learn from this interaction
        self._learn_from_execution(user_input, state)
        
        return state
    
    def _execute_enhanced_plan(self, plan: List[ExecutionStep], user_request: str, 
                             analysis: Dict[str, Any]) -> List[ToolResult]:
        """Execute plan with enhanced error handling, retries, and decision making"""
        import time
        results = []
        step_outputs = {}
        step_data = {}
        generated_content = {}

        def replace_placeholders(val: str) -> str:
            def repl(match):
                placeholder = match.group(1)
                if placeholder == "generated_content":
                    return generated_content.get("generated_content", "[ERROR: No generated content]")
                
                m = re.match(r"step_(\d+)\.output", placeholder)
                if m:
                    step_num = int(m.group(1))
                    return step_outputs.get(step_num, f"[ERROR: step_{step_num}.output not found]")
                
                m = re.match(r"step_(\d+)\.data\.([\w_]+)", placeholder)
                if m:
                    step_num = int(m.group(1))
                    key = m.group(2)
                    return str(step_data.get(step_num, {}).get(key, f"[ERROR: step_{step_num}.data.{key} not found]"))
                
                return f"[ERROR: Unresolved placeholder {{{{{placeholder}}}}}]"
            
            return re.sub(r"{{\s*([^{}]+)\s*}}", repl, val)

        def enhanced_llm_with_retry(request, content_type, step: ExecutionStep, max_retries=3):
            for attempt in range(max_retries):
                response = self.llm.generate_content(request, content_type, max_retries=1)
                
                if response and not response.content.startswith("Error generating content"):
                    return response
                
                # Decision point: Should we retry with different approach?
                if attempt < max_retries - 1:
                    decision = self.llm.make_decision(
                        DecisionType.ERROR_RECOVERY,
                        {
                            "error": response.content if response else "No response",
                            "attempt": attempt + 1,
                            "step_description": step.description
                        },
                        ["retry_same_approach", "try_fallback", "simplify_request"]
                    )
                    
                    if decision.decision == "simplify_request":
                        request = f"Simplified request: {request[:200]}"
                    elif decision.decision == "try_fallback" and step.fallback_actions:
                        # Try fallback action instead
                        break
                
                print(f"[WARN] LLM retry {attempt + 1}/{max_retries}: {response.content if response else 'No response'}")
                time.sleep(2 ** attempt)
            
            return response if response else LLMResponse(content="Error: All LLM attempts failed", confidence=0.0)

        for idx, step in enumerate(plan):
            print(f"  📌 Step {step.step}: {step.description}")
            
            # Check dependencies
            if step.depends_on:
                missing_deps = [dep for dep in step.depends_on if dep not in step_data]
                if missing_deps:
                    results.append(ToolResult(
                        False, "", f"Missing dependencies: {missing_deps}",
                        confidence=0.0, quality_score=0.0
                    ))
                    continue
            
            # Execute step with retries
            for attempt in range(step.max_retries):
                try:
                    # Replace placeholders in parameters
                    parameters = {}
                    for key, value in step.parameters.items():
                        if isinstance(value, str):
                            parameters[key] = replace_placeholders(value)
                        else:
                            parameters[key] = value
                    
                    result = None
                    
                    # Execute based on action type
                    if step.action == "web":
                        print('... WebTool use started ...')
                        tool = self.tools[step.action]
                        result = tool.execute(**parameters)
                        print('... WebTool use completed ...')
                    elif step.action == "filesystem":
                        print('... FileSystemTool use started ...')
                        tool = self.tools[step.action]
                        result = tool.execute(step.operation, **parameters)
                        print('... FileSystemTool use completed ...')
                    elif step.action=="data_processing":
                        print('... DataProcessingTool use started ...')
                        tool=self.tools[step.action]
                        if parameters.get('text',''):
                            result = tool.execute(step.operation,parameters.get('text',''),**parameters)
                        elif parameters.get('data',''):
                            print(parameters.get('data',''))
                            result = tool.execute(step.operation,parameters.get('data',''),**parameters)
                        else:
                            result = ToolResult(False, "", f"Parameter is missing : {step.action}")
                        print('... DataProcessingTool use completed ...')
                    elif step.action == "llm_generation":
                        print('... LLMGenerationTool use started ...')
                        tool = self.tools[step.action]
                        content_type = "code" if "code" in step.operation else "general"
                        if "creative" in user_request.lower():
                            content_type = "creative"
                        
                        llm_response = enhanced_llm_with_retry(
                            parameters.get("request", user_request), 
                            content_type, 
                            step
                        )
                        
                        generated_content["generated_content"] = llm_response.content
                        step_outputs[step.step] = llm_response.content
                        step_data[step.step] = {"content": llm_response.content}
                        
                        success = not llm_response.content.startswith("Error")
                        result = ToolResult(
                            success=success,
                            output=f"Content generated ({len(llm_response.content)} characters)" if success else llm_response.content,
                            error=None if success else llm_response.content,
                            data={"content": llm_response.content},
                            confidence=llm_response.confidence,
                            quality_score=0.8 if success else 0.0
                        )
                        print('... LLMGenerationTool use completed ...')
                    elif step.action in self.tools:
                        print('... Tool use started ...')
                        tool = self.tools[step.action]
                        result = tool.execute(step.operation, **parameters)
                        print('... Tool use completed ...')
                    
                        # Enhanced quality assessment
                        if hasattr(tool, 'assess_quality'):
                            result.quality_score = tool.assess_quality(result)
                    else:
                        result = ToolResult(False, "", f"Unknown action: {step.action}")
                    
                    # Check success criteria
                    success_met = self._check_success_criteria(result, step.success_criteria)
                    
                    if result.success and success_met:
                        step_outputs[step.step] = result.output
                        step_data[step.step] = result.data if result.data else {}
                        results.append(result)
                        break
                    elif attempt < step.max_retries - 1:
                        # Try fallback actions
                        if step.fallback_actions and attempt == 0:
                            fallback_action = step.fallback_actions[0]
                            step.action = fallback_action.get("action", step.action)
                            step.operation = fallback_action.get("operation", step.operation)
                            step.parameters = fallback_action.get("parameters", step.parameters)
                            print(f"  🔄 Using fallback approach for step {step.step}")
                        else:
                            print(f"  ⚠️  Retrying step {step.step} (attempt {attempt + 2}/{step.max_retries})")
                            time.sleep(1)
                    else:
                        # All retries exhausted
                        results.append(result)
                        step_outputs[step.step] = result.output
                        step_data[step.step] = result.data if result.data else {}
                        break
                        
                except Exception as e:
                    error_result = ToolResult(False, "", f"Step execution failed: {str(e)}")
                    if attempt == step.max_retries - 1:
                        results.append(error_result)
                        break
                    else:
                        print(f"  ❌ Step {step.step} error (attempt {attempt + 1}): {str(e)}")
                        time.sleep(1)
        
        return results
    
    def _check_success_criteria(self, result: ToolResult, criteria: List[str]) -> bool:
        """Check if result meets success criteria"""
        if not criteria:
            return result.success
        
        # Simple criteria checking - can be enhanced with more sophisticated logic
        for criterion in criteria:
            if "generated" in criterion.lower() and not result.output:
                return False
            elif "file" in criterion.lower() and "created" in criterion.lower():
                if not result.success or "created" not in result.output.lower():
                    return False
        
        return result.success
    
    def _assess_execution_quality(self, results: List[ToolResult], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall execution quality"""
        if not results:
            return {"overall_quality": 0.0, "assessment": "No results to assess"}
        
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_quality = sum(r.quality_score for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Weighted overall quality
        overall_quality = (success_rate * 0.4) + (avg_quality * 0.4) + (avg_confidence * 0.2)
        
        assessment = {
            "overall_quality": overall_quality,
            "success_rate": success_rate,
            "average_quality_score": avg_quality,
            "average_confidence": avg_confidence,
            "total_steps": len(results),
            "successful_steps": sum(1 for r in results if r.success),
            "assessment": self._generate_quality_assessment(overall_quality)
        }
        
        return assessment
    
    def _generate_quality_assessment(self, quality_score: float) -> str:
        """Generate textual quality assessment"""
        if quality_score >= 0.9:
            return "Excellent execution with high quality results"
        elif quality_score >= 0.7:
            return "Good execution with satisfactory results"
        elif quality_score >= 0.5:
            return "Acceptable execution with some issues"
        elif quality_score >= 0.3:
            return "Poor execution with significant problems"
        else:
            return "Failed execution with critical issues"
    
    def _generate_enhanced_response(self, analysis: Dict[str, Any], plan: List[ExecutionStep], 
                                  results: List[ToolResult], user_request: str, state: Dict[str, Any]) -> str:
        """Generate comprehensive response with quality metrics and insights"""
        
        response_parts = []
        
        # Header with quality indicator
        quality_assessment = state.get("quality_assessment", {})
        overall_quality = quality_assessment.get("overall_quality", 0.5)
        
        if overall_quality >= 0.8:
            icon = "🌟"
            status = "Excellent"
        elif overall_quality >= 0.6:
            icon = "✅"
            status = "Good"
        elif overall_quality >= 0.4:
            icon = "⚠️"
            status = "Acceptable"
        else:
            icon = "❌"
            status = "Needs Improvement"
        
        task_type = analysis.get("primary_task_type", "general")
        complexity = analysis.get("complexity", "medium")
        
        response_parts.append(f"{icon} {status} Execution Complete!")
        response_parts.append(f"Task: {task_type.title().replace('_', ' ')} | Complexity: {complexity.title()} | Quality: {overall_quality:.1%}")
        
        # Execution summary
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        
        if success_count == total_count and total_count > 0:
            response_parts.append(f"\n🎯 All {total_count} steps completed successfully")
        elif success_count > 0:
            response_parts.append(f"\n⚡ Completed {success_count}/{total_count} steps")
        else:
            response_parts.append(f"\n❌ Task execution failed")
        
        # Show accomplishments
        accomplishments = self._extract_accomplishments(plan, results)
        if accomplishments:
            response_parts.append(f"\n📋 Accomplishments:")
            for acc in accomplishments:
                response_parts.append(f"   • {acc}")
        
        # Decision insights
        decisions_made = state.get("decisions_made", [])
        if decisions_made:
            response_parts.append(f"\n🧠 Key Decisions Made:")
            for decision in decisions_made[-3:]:  # Show last 3 decisions
                response_parts.append(f"   • {decision.decision_type.value}: {decision.decision}")
        
        # Quality insights
        if quality_assessment:
            response_parts.append(f"\n📊 Quality Metrics:")
            response_parts.append(f"   • Success Rate: {quality_assessment.get('success_rate', 0):.1%}")
            response_parts.append(f"   • Average Quality: {quality_assessment.get('average_quality_score', 0):.1%}")
            response_parts.append(f"   • Confidence: {quality_assessment.get('average_confidence', 0):.1%}")
        
        # Execution details
        response_parts.append(f"\n📝 Execution Details:")
        for i, (step, result) in enumerate(zip(plan, results)):
            status = "✅" if result.success else "❌"
            quality_indicator = ""
            if hasattr(result, 'quality_score'):
                if result.quality_score >= 0.8:
                    quality_indicator = "🔥"
                elif result.quality_score >= 0.6:
                    quality_indicator = "👍"
                elif result.quality_score < 0.4:
                    quality_indicator = "⚠️"
            
            response_parts.append(f"  {status} {step.description} {quality_indicator}")
            
            if not result.success and result.error:
                response_parts.append(f"      Issue: {result.error[:100]}...")
        
        # Performance insights
        iterations = len(state.get("history", []))
        if iterations > 1:
            response_parts.append(f"\n🔄 Required {iterations} iterations for optimal results")
        
        # Content output (for content generation tasks)
        content_results = [r for r in results if r.data and 'content' in r.data]
        if content_results and task_type in ['creative_writing', 'code_generation']:
            latest_content = content_results[-1].data['content']
            if len(latest_content) > 500:
                response_parts.append(f"\n📄 Generated Content Preview:")
                response_parts.append(f"{latest_content[:200]}...")
                response_parts.append(f"\n(Full content: {len(latest_content)} characters)")
            else:
                response_parts.append(f"\n📄 Generated Content:")
                response_parts.append(f"{latest_content}")
        
        # Next steps or recommendations
        if overall_quality < 0.7:
            response_parts.append(f"\n💡 Recommendations for improvement:")
            if quality_assessment.get('success_rate', 1) < 0.8:
                response_parts.append(f"   • Review failed steps and error messages")
            if quality_assessment.get('average_confidence', 1) < 0.7:
                response_parts.append(f"   • Consider providing more specific requirements")
            if complexity == "expert" and overall_quality < 0.6:
                response_parts.append(f"   • Complex tasks may benefit from step-by-step breakdown")
        
        return '\n'.join(response_parts)
    
    def _extract_accomplishments(self, plan: List[ExecutionStep], results: List[ToolResult]) -> List[str]:
        """Extract key accomplishments from execution results"""
        accomplishments = []
        
        for step, result in zip(plan, results):
            if result.success:
                if step.action == "filesystem":
                    if step.operation == "create_file":
                        accomplishments.append(f"Created file: {step.parameters.get('path', 'unknown')}")
                    elif step.operation == "write_file":
                        accomplishments.append(f"Updated file: {step.parameters.get('path', 'unknown')}")
                    elif step.operation == "read_file":
                        accomplishments.append(f"Successfully read file: {step.parameters.get('path', 'unknown')}")
                elif step.action == "edit_file":
                    accomplishments.append(f"Edited file: {step.parameters.get('file_path', 'unknown')}")
                elif step.action == "command":
                    accomplishments.append(f"Executed command: {step.parameters.get('command', 'unknown')[:50]}...")
                elif step.action == "web":
                    accomplishments.append(f"Completed web research on: {step.parameters.get('data', 'topic')}")
                elif step.action == "llm_generation":
                    content_length = len(result.data.get('content', '')) if result.data else 0
                    accomplishments.append(f"Generated content ({content_length} characters)")
                elif step.action == "data_processing":
                    accomplishments.append(f"Processed data using {step.operation}")
        
        return accomplishments
    
    def _learn_from_execution(self, user_input: str, state: Dict[str, Any]) -> None:
        """Learn from execution patterns and user preferences"""
        try:
            # Extract patterns from successful executions
            if state.get("goal_met", False):
                task_type = state["history"][-1]["analysis"].get("primary_task_type", "general") if state["history"] else "general"
                complexity = state["history"][-1]["analysis"].get("complexity", "medium") if state["history"] else "medium"
                
                # Update success patterns
                pattern_key = f"{task_type}_{complexity}"
                if pattern_key not in self.common_patterns:
                    self.common_patterns[pattern_key] = {
                        "count": 0,
                        "avg_quality": 0,
                        "avg_iterations": 0,
                        "common_tools": {}
                    }
                
                pattern = self.common_patterns[pattern_key]
                pattern["count"] += 1
                
                # Update quality metrics
                quality = state.get("quality_assessment", {}).get("overall_quality", 0.5)
                pattern["avg_quality"] = (pattern["avg_quality"] * (pattern["count"] - 1) + quality) / pattern["count"]
                
                # Update iteration metrics
                iterations = len(state.get("history", []))
                pattern["avg_iterations"] = (pattern["avg_iterations"] * (pattern["count"] - 1) + iterations) / pattern["count"]
                
                # Track tool usage
                for hist in state.get("history", []):
                    for step in hist.get("plan", []):
                        tool = step.action
                        pattern["common_tools"][tool] = pattern["common_tools"].get(tool, 0) + 1
            
            # Learn from errors
            else:
                error_patterns = []
                for hist in state.get("history", []):
                    for result in hist.get("results", []):
                        if not result.success and result.error:
                            error_patterns.append(result.error[:100])
                
                if error_patterns:
                    error_key = str(hash(str(error_patterns)))[:10]
                    self.error_patterns[error_key] = {
                        "patterns": error_patterns,
                        "count": self.error_patterns.get(error_key, {}).get("count", 0) + 1,
                        "context": user_input[:100]
                    }
            
            # Learn user preferences from requests
            request_lower = user_input.lower()
            if "format" in request_lower or "style" in request_lower:
                self.user_preferences["formatting_preference"] = True
            if "detailed" in request_lower or "comprehensive" in request_lower:
                self.user_preferences["detail_preference"] = "high"
            if "simple" in request_lower or "basic" in request_lower:
                self.user_preferences["detail_preference"] = "low"
                
        except Exception as e:
            print(f"Learning error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        metrics = self.performance_metrics.copy()
        
        # Calculate additional metrics
        if metrics["total_tasks"] > 0:
            metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"]
        else:
            metrics["success_rate"] = 0.0
        
        if self.performance_metrics["quality_scores"]:
            metrics["average_quality"] = sum(self.performance_metrics["quality_scores"]) / len(self.performance_metrics["quality_scores"])
            metrics["quality_trend"] = "improving" if len(self.performance_metrics["quality_scores"]) > 5 and \
                sum(self.performance_metrics["quality_scores"][-3:]) / 3 > sum(self.performance_metrics["quality_scores"][:3]) / 3 else "stable"
        else:
            metrics["average_quality"] = 0.0
            metrics["quality_trend"] = "unknown"
        
        # Add learning insights
        metrics["learned_patterns"] = len(self.common_patterns)
        metrics["error_patterns"] = len(self.error_patterns)
        metrics["user_preferences"] = self.user_preferences.copy()
        
        # Most successful task types
        if self.common_patterns:
            best_pattern = max(self.common_patterns.items(), key=lambda x: x[1]["avg_quality"])
            metrics["best_task_type"] = best_pattern[0]
            metrics["best_task_quality"] = best_pattern[1]["avg_quality"]
        
        return metrics
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving agent performance"""
        recommendations = []
        metrics = self.get_performance_summary()
        
        # Success rate recommendations
        if metrics["success_rate"] < 0.7:
            recommendations.append("Consider providing more detailed task specifications")
            recommendations.append("Break complex tasks into smaller, manageable steps")
        
        # Quality recommendations
        if metrics["average_quality"] < 0.6:
            recommendations.append("Review error patterns and common failure points")
            recommendations.append("Consider adjusting LLM parameters for better output quality")
        
        # Performance recommendations
        if metrics["average_execution_time"] > 60:
            recommendations.append("Optimize tool selection for faster execution")
            recommendations.append("Consider parallel execution for independent steps")
        
        # Learning recommendations
        if metrics["learned_patterns"] < 5:
            recommendations.append("Increase task variety to improve pattern recognition")
        
        # Error pattern recommendations
        if metrics["error_patterns"] > 10:
            recommendations.append("Implement better error handling and recovery mechanisms")
        
        return recommendations
    
    def reset_learning(self) -> None:
        """Reset learning data while preserving performance metrics"""
        self.common_patterns.clear()
        self.error_patterns.clear()
        self.user_preferences.clear()
        print("Learning data reset. Performance metrics preserved.")
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis or transfer"""
        return {
            "common_patterns": self.common_patterns,
            "error_patterns": self.error_patterns,
            "user_preferences": self.user_preferences,
            "performance_metrics": self.performance_metrics,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_learning_data(self, learning_data: Dict[str, Any]) -> bool:
        """Import learning data from previous sessions"""
        try:
            if "common_patterns" in learning_data:
                self.common_patterns.update(learning_data["common_patterns"])
            if "error_patterns" in learning_data:
                self.error_patterns.update(learning_data["error_patterns"])
            if "user_preferences" in learning_data:
                self.user_preferences.update(learning_data["user_preferences"])
            if "performance_metrics" in learning_data:
                # Merge performance metrics carefully
                imported_metrics = learning_data["performance_metrics"]
                self.performance_metrics["total_tasks"] += imported_metrics.get("total_tasks", 0)
                self.performance_metrics["successful_tasks"] += imported_metrics.get("successful_tasks", 0)
                self.performance_metrics["quality_scores"].extend(imported_metrics.get("quality_scores", []))
            
            print(f"Learning data imported successfully from {learning_data.get('export_timestamp', 'unknown time')}")
            return True
        except Exception as e:
            print(f"Failed to import learning data: {e}")
            return False

# Create alias for backward compatibility
GeneralGroqAgent = EnhancedGeneralGroqAgent

# Usage example and testing functions
def create_enhanced_agent(groq_api_key: str) -> EnhancedGeneralGroqAgent:
    """Create and configure an enhanced agent instance"""
    return EnhancedGeneralGroqAgent(groq_api_key)

def run_agent_with_monitoring(agent: EnhancedGeneralGroqAgent, user_request: str) -> Dict[str, Any]:
    """Run agent with comprehensive monitoring and logging"""
    print(f"\n🚀 Processing Request: {user_request}")
    print("=" * 80)
    
    start_time = datetime.now()
    try:
        result = agent.process_request(user_request)
        end_time = datetime.now()
        
        print(f"\n📊 Execution Summary:")
        print(f"   Duration: {(end_time - start_time).total_seconds():.2f}s")
        print(f"   Goal Met: {'✅' if result.get('goal_met', False) else '❌'}")
        print(f"   Iterations: {len(result.get('history', []))}")
        print(f"   Decisions Made: {len(result.get('decisions_made', []))}")
        
        quality_assessment = result.get('quality_assessment', {})
        if quality_assessment:
            print(f"   Overall Quality: {quality_assessment.get('overall_quality', 0):.1%}")
            print(f"   Success Rate: {quality_assessment.get('success_rate', 0):.1%}")
        
        return result
        
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return {"error": str(e), "goal_met": False}

def demo_enhanced_features():
    """Demonstrate enhanced agent features"""
    print("🔬 Enhanced Groq Agent Demo")
    print("=" * 50)
    
    # This would require actual API key
    print("Features demonstrated:")
    print("✅ Enhanced decision making with multiple decision types")
    print("✅ Quality assessment and adaptive execution")
    print("✅ Learning from patterns and user preferences")
    print("✅ Comprehensive error handling and recovery")
    print("✅ Performance monitoring and recommendations")
    print("✅ Multi-step planning with dependencies")
    print("✅ Context-aware tool selection")
    print("✅ Fallback strategies for failed operations")
    
    # Mock example of what the enhanced features provide
    mock_performance = {
        "total_tasks": 25,
        "successful_tasks": 22,
        "success_rate": 0.88,
        "average_quality": 0.75,
        "quality_trend": "improving",
        "learned_patterns": 8,
        "best_task_type": "code_generation_medium",
        "best_task_quality": 0.92
    }
    
    print(f"\n📈 Sample Performance Metrics:")
    for key, value in mock_performance.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    agent=EnhancedGeneralGroqAgent(os.getenv("GROQ_API_KEY"))
    print(run_agent_with_monitoring(agent,"Analyse the Webtool.py and Save it into a summary.txt for it."))