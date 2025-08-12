# agent.py - General Coding Agent Implementation
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess

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
class CodeRequest:
    """Structure for code generation requests"""
    language: str
    project_type: str
    description: str
    features: List[str]
    file_name: str
    project_name: str

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
        return "Create, read, write, and manage files and directories"
    
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
            elif operation == "delete_file":
                return self._delete_file(kwargs.get('path'))
            elif operation == "create_project_env":
                return self._create_project_environment(kwargs.get('project_name'), kwargs.get('language'))
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_file(self, path: str, content: str = '') -> ToolResult:
        """Create a new file with optional content"""
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(True, f"File created successfully: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_directory(self, path: str) -> ToolResult:
        """Create a directory"""
        try:
            os.makedirs(path, exist_ok=True)
            return ToolResult(True, f"Directory created: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _read_file(self, path: str) -> ToolResult:
        """Read file content"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(True, content, data={"content": content, "path": path})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(True, f"Content written to: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _list_directory(self, path: str) -> ToolResult:
        """List directory contents"""
        try:
            items = os.listdir(path)
            return ToolResult(True, f"Directory contents: {', '.join(items)}", data={"items": items})
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _delete_file(self, path: str) -> ToolResult:
        """Delete a file"""
        try:
            os.remove(path)
            return ToolResult(True, f"File deleted: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_project_environment(self, project_name: str, language: str) -> ToolResult:
        """Create appropriate project environment based on language"""
        try:
            base_path = os.path.join(".", project_name)
            os.makedirs(base_path, exist_ok=True)
            
            created_files = []
            
            if language.lower() in ["python", "py"]:
                # Python environment
                files = {
                    "requirements.txt": "",
                    "README.md": f"# {project_name.title()}\n\n## Installation\n```bash\npip install -r requirements.txt\n```\n\n## Usage\n```bash\npython main.py\n```",
                    ".gitignore": "__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\nenv/\nvenv/\n.env\n.venv/\ndist/\nbuild/\n*.egg-info/",
                    "src/__init__.py": "",
                    "tests/__init__.py": ""
                }
            
            elif language.lower() in ["javascript", "js", "node"]:
                # Node.js environment
                files = {
                    "package.json": f'{{\n  "name": "{project_name.lower().replace(" ", "-")}",\n  "version": "1.0.0",\n  "description": "",\n  "main": "index.js",\n  "scripts": {{\n    "start": "node index.js",\n    "test": "echo \\"Error: no test specified\\" && exit 1"\n  }},\n  "keywords": [],\n  "author": "",\n  "license": "ISC"\n}}',
                    "README.md": f"# {project_name.title()}\n\n## Installation\n```bash\nnpm install\n```\n\n## Usage\n```bash\nnpm start\n```",
                    ".gitignore": "node_modules/\n.npm\n.env\ndist/\nbuild/"
                }
            
            elif language.lower() in ["html", "web", "css"]:
                # Web environment
                files = {
                    "index.html": f"<!DOCTYPE html>\n<html>\n<head>\n    <title>{project_name.title()}</title>\n    <link rel=\"stylesheet\" href=\"styles.css\">\n</head>\n<body>\n    <h1>{project_name.title()}</h1>\n    <script src=\"script.js\"></script>\n</body>\n</html>",
                    "styles.css": f"/* Styles for {project_name} */\nbody {{ font-family: Arial, sans-serif; }}",
                    "script.js": f"// JavaScript for {project_name}\nconsole.log('Loaded!');",
                    "README.md": f"# {project_name.title()}\n\nOpen index.html in your browser."
                }
            
            elif language.lower() in ["java"]:
                # Java environment
                files = {
                    f"src/main/java/Main.java": f"public class Main {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello from {project_name}!\");\n    }}\n}}",
                    "README.md": f"# {project_name.title()}\n\n## Compilation\n```bash\njavac src/main/java/Main.java\n```\n\n## Usage\n```bash\njava -cp src/main/java Main\n```",
                    ".gitignore": "*.class\n*.jar\ntarget/\nbuild/"
                }
            
            else:
                # Generic environment
                files = {
                    "README.md": f"# {project_name.title()}\n\nProject created for {language}"
                }
            
            # Create all files
            for file_path, content in files.items():
                full_path = os.path.join(base_path, file_path)
                dir_path = os.path.dirname(full_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_files.append(file_path)
            
            return ToolResult(True, f"Project environment created with files: {', '.join(created_files)}")
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
        """Execute system command"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=kwargs.get('working_directory', '.')
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nErrors: {result.stderr}"
            
            success = result.returncode == 0
            return ToolResult(success, output, None if success else result.stderr)
        except Exception as e:
            return ToolResult(False, "", str(e))

class CodeGenerator:
    """Intelligent code generator that creates code based on natural language descriptions"""
    
    def __init__(self):
        self.language_extensions = {
            'python': '.py',
            'javascript': '.js',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c',
            'html': '.html',
            'css': '.css',
            'php': '.php',
            'ruby': '.rb',
            'go': '.go',
            'rust': '.rs'
        }
        
        self.language_templates = {
            'python': {
                'header': '#!/usr/bin/env python3\n"""',
                'imports': 'import os\nimport sys\nfrom typing import List, Dict, Optional',
                'main_function': 'def main():\n    pass\n\nif __name__ == "__main__":\n    main()',
                'class_template': 'class {class_name}:\n    def __init__(self):\n        pass'
            },
            'javascript': {
                'header': '/**',
                'imports': '// Add imports here',
                'main_function': 'function main() {\n    console.log("Hello World!");\n}\n\nmain();',
                'class_template': 'class {class_name} {\n    constructor() {\n        // Constructor\n    }\n}'
            },
            'java': {
                'header': '/**',
                'imports': 'import java.util.*;',
                'main_function': 'public static void main(String[] args) {\n        System.out.println("Hello World!");\n    }',
                'class_template': 'public class {class_name} {\n    public {class_name}() {\n        // Constructor\n    }\n}'
            }
        }
    
    def generate_code_from_description(self, code_request: CodeRequest) -> str:
        """Generate code based on natural language description"""
        language = code_request.language.lower()
        description = code_request.description.lower()
        
        # Analyze the description to understand what to build
        code_structure = self._analyze_code_requirements(description, code_request.features)
        
        # Generate the actual code
        generated_code = self._build_code(language, code_structure, code_request)
        
        return generated_code
    
    def _analyze_code_requirements(self, description: str, features: List[str]) -> Dict[str, Any]:
        """Analyze the description to determine code structure"""
        structure = {
            'needs_classes': False,
            'needs_functions': [],
            'needs_main': True,
            'imports': [],
            'class_names': [],
            'application_type': 'console',
            'complexity': 'simple'
        }
        
        # Detect if classes are needed
        class_indicators = ['class', 'object', 'oop', 'inheritance', 'method']
        if any(indicator in description for indicator in class_indicators):
            structure['needs_classes'] = True
        
        # Detect application type
        if any(term in description for term in ['game', 'chess', 'tic tac toe', 'puzzle']):
            structure['application_type'] = 'game'
            structure['needs_classes'] = True
            structure['complexity'] = 'medium'
        
        elif any(term in description for term in ['calculator', 'math', 'compute']):
            structure['application_type'] = 'calculator'
            structure['needs_functions'] = ['add', 'subtract', 'multiply', 'divide']
        
        elif any(term in description for term in ['todo', 'task', 'list', 'manage']):
            structure['application_type'] = 'todo'
            structure['needs_classes'] = True
            structure['complexity'] = 'medium'
        
        elif any(term in description for term in ['web', 'scrape', 'crawl', 'requests']):
            structure['application_type'] = 'web_scraper'
            structure['imports'] = ['requests', 'bs4']
            
        elif any(term in description for term in ['api', 'rest', 'http', 'client']):
            structure['application_type'] = 'api_client'
            structure['imports'] = ['requests', 'json']
        
        elif any(term in description for term in ['gui', 'window', 'interface', 'tkinter']):
            structure['application_type'] = 'gui'
            structure['imports'] = ['tkinter']
            structure['needs_classes'] = True
        
        # Extract class names from description
        words = description.split()
        for i, word in enumerate(words):
            if word in ['class', 'object'] and i + 1 < len(words):
                class_name = words[i + 1].capitalize()
                if class_name not in structure['class_names']:
                    structure['class_names'].append(class_name)
        
        return structure
    
    def _build_code(self, language: str, structure: Dict[str, Any], code_request: CodeRequest) -> str:
        """Build the actual code based on structure analysis"""
        if language not in self.language_templates:
            return self._generate_generic_code(language, code_request)
        
        template = self.language_templates[language]
        code_parts = []
        
        # Add header and description
        if language == 'python':
            code_parts.append(f'{template["header"]}\n{code_request.description}\n"""')
        else:
            code_parts.append(f'{template["header"]}\n * {code_request.description}\n */')
        
        code_parts.append("")  # Empty line
        
        # Add imports
        imports = self._generate_imports(language, structure['imports'])
        if imports:
            code_parts.append(imports)
            code_parts.append("")
        
        # Generate classes if needed
        if structure['needs_classes']:
            if not structure['class_names']:
                # Generate class name from project name
                class_name = ''.join(word.capitalize() for word in code_request.project_name.split('_'))
                structure['class_names'] = [class_name]
            
            for class_name in structure['class_names']:
                class_code = self._generate_class(language, class_name, structure)
                code_parts.append(class_code)
                code_parts.append("")
        
        # Generate functions if needed
        for func_name in structure['needs_functions']:
            func_code = self._generate_function(language, func_name)
            code_parts.append(func_code)
            code_parts.append("")
        
        # Add main function/code
        if structure['needs_main']:
            main_code = self._generate_main_code(language, structure, code_request)
            code_parts.append(main_code)
        
        return '\n'.join(code_parts)
    
    def _generate_imports(self, language: str, imports: List[str]) -> str:
        """Generate import statements"""
        if not imports:
            return ""
        
        if language == 'python':
            import_lines = []
            for imp in imports:
                if imp in ['requests', 'json', 'os', 'sys', 'tkinter']:
                    import_lines.append(f"import {imp}")
                elif imp == 'bs4':
                    import_lines.append("from bs4 import BeautifulSoup")
                else:
                    import_lines.append(f"import {imp}")
            return '\n'.join(import_lines)
        
        elif language == 'javascript':
            # For Node.js
            import_lines = []
            for imp in imports:
                import_lines.append(f"const {imp} = require('{imp}');")
            return '\n'.join(import_lines)
        
        return ""
    
    def _generate_class(self, language: str, class_name: str, structure: Dict[str, Any]) -> str:
        """Generate a class based on application type"""
        if language == 'python':
            class_code = f"class {class_name}:\n    def __init__(self):\n"
            
            # Add attributes based on application type
            if structure['application_type'] == 'game':
                class_code += "        self.game_state = 'playing'\n"
                class_code += "        self.player_turn = 1\n"
                class_code += "        self.board = []\n"
                class_code += "\n    def start_game(self):\n"
                class_code += "        \"\"\"Initialize and start the game\"\"\"\n"
                class_code += "        print('Game started!')\n"
                class_code += "\n    def make_move(self, move):\n"
                class_code += "        \"\"\"Process a player move\"\"\"\n"
                class_code += "        pass\n"
                class_code += "\n    def check_winner(self):\n"
                class_code += "        \"\"\"Check if there's a winner\"\"\"\n"
                class_code += "        return None"
                
            elif structure['application_type'] == 'todo':
                class_code += "        self.tasks = []\n"
                class_code += "        self.next_id = 1\n"
                class_code += "\n    def add_task(self, description):\n"
                class_code += "        \"\"\"Add a new task\"\"\"\n"
                class_code += "        task = {'id': self.next_id, 'description': description, 'completed': False}\n"
                class_code += "        self.tasks.append(task)\n"
                class_code += "        self.next_id += 1\n"
                class_code += "\n    def complete_task(self, task_id):\n"
                class_code += "        \"\"\"Mark task as completed\"\"\"\n"
                class_code += "        for task in self.tasks:\n"
                class_code += "            if task['id'] == task_id:\n"
                class_code += "                task['completed'] = True\n"
                class_code += "                break\n"
                class_code += "\n    def list_tasks(self):\n"
                class_code += "        \"\"\"Display all tasks\"\"\"\n"
                class_code += "        for task in self.tasks:\n"
                class_code += "            status = '✓' if task['completed'] else '○'\n"
                class_code += "            print(f\"{status} {task['id']}: {task['description']}\")"
            
            else:
                # Generic class
                class_code += "        pass\n"
                class_code += "\n    def run(self):\n"
                class_code += "        \"\"\"Main method to run the application\"\"\"\n"
                class_code += "        print(f'{class_name} is running!')"
            
            return class_code
        
        elif language == 'javascript':
            class_code = f"class {class_name} {{\n    constructor() {{\n"
            
            if structure['application_type'] == 'game':
                class_code += "        this.gameState = 'playing';\n"
                class_code += "        this.playerTurn = 1;\n"
                class_code += "        this.board = [];\n"
                class_code += "    }\n\n"
                class_code += "    startGame() {\n"
                class_code += "        console.log('Game started!');\n"
                class_code += "    }\n\n"
                class_code += "    makeMove(move) {\n"
                class_code += "        // Process player move\n"
                class_code += "    }\n"
            else:
                class_code += "        // Initialize properties\n"
                class_code += "    }\n\n"
                class_code += "    run() {\n"
                class_code += "        console.log(`${this.constructor.name} is running!`);\n"
                class_code += "    }\n"
            
            class_code += "}"
            return class_code
        
        return f"// {class_name} class would be implemented here"
    
    def _generate_function(self, language: str, func_name: str) -> str:
        """Generate a function based on its name"""
        if language == 'python':
            if func_name in ['add', 'subtract', 'multiply', 'divide']:
                return f"def {func_name}(a, b):\n    \"\"\"Perform {func_name} operation\"\"\"\n    {'return a + b' if func_name == 'add' else 'return a - b' if func_name == 'subtract' else 'return a * b' if func_name == 'multiply' else 'return a / b if b != 0 else None'}"
            else:
                return f"def {func_name}():\n    \"\"\"Function: {func_name}\"\"\"\n    pass"
        
        elif language == 'javascript':
            if func_name in ['add', 'subtract', 'multiply', 'divide']:
                op = '+' if func_name == 'add' else '-' if func_name == 'subtract' else '*' if func_name == 'multiply' else '/'
                return f"function {func_name}(a, b) {{\n    return a {op} b;\n}}"
            else:
                return f"function {func_name}() {{\n    // Implement {func_name}\n}}"
        
        return f"// {func_name} function"
    
    def _generate_main_code(self, language: str, structure: Dict[str, Any], code_request: CodeRequest) -> str:
        """Generate main execution code"""
        if language == 'python':
            if structure['needs_classes'] and structure['class_names']:
                class_name = structure['class_names'][0]
                return f"def main():\n    \"\"\"Main function\"\"\"\n    app = {class_name}()\n    app.run()\n\nif __name__ == '__main__':\n    main()"
            else:
                return "def main():\n    \"\"\"Main function\"\"\"\n    print('Hello, World!')\n    # Add your code here\n\nif __name__ == '__main__':\n    main()"
        
        elif language == 'javascript':
            if structure['needs_classes'] and structure['class_names']:
                class_name = structure['class_names'][0]
                return f"function main() {{\n    const app = new {class_name}();\n    app.run();\n}}\n\nmain();"
            else:
                return "function main() {\n    console.log('Hello, World!');\n    // Add your code here\n}\n\nmain();"
        
        return "// Main code here"
    
    def _generate_generic_code(self, language: str, code_request: CodeRequest) -> str:
        """Generate generic code for unsupported languages"""
        ext = self.language_extensions.get(language, '.txt')
        comment_char = '//' if ext in ['.js', '.java', '.cpp', '.c'] else '#'
        
        return f"""{comment_char} {code_request.description}
{comment_char} Generated for {language}
{comment_char} Project: {code_request.project_name}

{comment_char} TODO: Implement your {code_request.project_type} here
{comment_char} Features requested: {', '.join(code_request.features)}
"""

class RequestAnalyzer:
    """Analyzes user requests and extracts coding requirements"""
    
    def __init__(self):
        self.language_patterns = {
            'python': ['python', 'py', 'django', 'flask', 'pandas'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue'],
            'java': ['java', 'spring', 'android'],
            'cpp': ['c++', 'cpp', 'cplus'],
            'c': ['c language', ' c '],
            'html': ['html', 'web page', 'website'],
            'css': ['css', 'styling', 'styles'],
            'php': ['php', 'laravel'],
            'ruby': ['ruby', 'rails'],
            'go': ['golang', 'go lang'],
            'rust': ['rust']
        }
        
        self.project_patterns = {
            'game': ['game', 'chess', 'tic tac toe', 'puzzle', 'arcade'],
            'calculator': ['calculator', 'math', 'arithmetic', 'compute'],
            'todo': ['todo', 'task manager', 'task list', 'organizer'],
            'web_scraper': ['scraper', 'scrape', 'crawl', 'extract data'],
            'api_client': ['api', 'rest client', 'http client'],
            'gui_app': ['gui', 'interface', 'window', 'desktop app'],
            'utility': ['tool', 'utility', 'helper', 'script'],
            'web_app': ['web app', 'website', 'web application']
        }
    
    def analyze_request(self, user_input: str) -> CodeRequest:
        """Analyze user request and return structured code request"""
        user_input_lower = user_input.lower()
        
        # Detect language
        detected_language = self._detect_language(user_input_lower)
        
        # Detect project type
        detected_project_type = self._detect_project_type(user_input_lower)
        
        # Extract project name
        project_name = self._extract_project_name(user_input)
        
        # Extract features
        features = self._extract_features(user_input_lower)
        
        # Generate file name
        file_name = self._generate_filename(project_name, detected_language, detected_project_type)
        
        return CodeRequest(
            language=detected_language,
            project_type=detected_project_type,
            description=user_input,
            features=features,
            file_name=file_name,
            project_name=project_name
        )
    
    def _detect_language(self, user_input: str) -> str:
        """Detect programming language from user input"""
        for language, patterns in self.language_patterns.items():
            if any(pattern in user_input for pattern in patterns):
                return language
        return 'python'  # Default to Python
    
    def _detect_project_type(self, user_input: str) -> str:
        """Detect project type from user input"""
        for project_type, patterns in self.project_patterns.items():
            if any(pattern in user_input for pattern in patterns):
                return project_type
        return 'utility'  # Default type
    
    def _extract_project_name(self, user_input: str) -> str:
        """Extract or generate project name"""
        # Look for patterns like "create a project called X" or "make an X app"
        patterns = [
            r'(?:project|app|program)\s+(?:called|named)\s+([a-zA-Z_]\w*)',
            r'(?:create|make|build)\s+(?:a|an)?\s*([a-zA-Z_]\w*)\s+(?:game|app|tool|program)',
            r'([a-zA-Z_]\w*)\s+(?:game|app|tool|program|project)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1).replace(' ', '_')
        
        # Generate name based on content
        words = user_input.lower().split()
        important_words = [w for w in words if w not in ['create', 'make', 'build', 'a', 'an', 'the', 'for', 'in', 'file']]
        
        if important_words:
            return '_'.join(important_words[:2])
        
        return 'my_project'
    
    def _extract_features(self, user_input: str) -> List[str]:
        """Extract features from user input"""
        features = []
        
        feature_keywords = {
            'gui': ['gui', 'interface', 'window'],
            'database': ['database', 'db', 'sqlite', 'mysql'],
            'networking': ['network', 'socket', 'client', 'server'],
            'file_io': ['file', 'read', 'write', 'save', 'load'],
            'graphics': ['graphics', 'drawing', 'animation'],
            'audio': ['audio', 'sound', 'music'],
            'web': ['web', 'http', 'api', 'rest']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _generate_filename(self, project_name: str, language: str, project_type: str) -> str:
        """Generate appropriate filename"""
        extension_map = {
            'python': '.py',
            'javascript': '.js',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c',
            'html': '.html',
            'css': '.css',
            'php': '.php',
            'ruby': '.rb',
            'go': '.go',
            'rust': '.rs'
        }
        
        extension = extension_map.get(language, '.txt')
        
        # Use project type for main filename if appropriate
        if project_type in ['game', 'calculator', 'todo']:
            base_name = project_type
        else:
            base_name = 'main'
        
        return f"{base_name}{extension}"

class TaskPlanner:
    """Plans and breaks down user requests into executable tasks"""
    
    def __init__(self):
        self.analyzer = RequestAnalyzer()
    
    def plan_tasks(self, user_input: str) -> List[Dict[str, Any]]:
        """Plan tasks based on user input"""
        code_request = self.analyzer.analyze_request(user_input)
        
        tasks = []
        
        # Task 1: Create project environment
        tasks.append({
            'type': 'create_environment',
            'code_request': code_request,
            'description': f"Create {code_request.language} environment for {code_request.project_name}"
        })
        
        # Task 2: Generate and create main code file
        tasks.append({
            'type': 'generate_code',
            'code_request': code_request,
            'description': f"Generate {code_request.project_type} code in {code_request.language}"
        })
        
        # Task 3: Create additional files if needed
        if code_request.features:
            tasks.append({
                'type': 'create_additional_files',
                'code_request': code_request,
                'description': f"Create additional files for features: {', '.join(code_request.features)}"
            })
        
        return tasks

class CodingAgent:
    """Main coding agent that can create any type of code project"""
    
    def __init__(self):
        self.tools = {
            'filesystem': FileSystemTool(),
            'command': CommandTool()
        }
        self.planner = TaskPlanner()
        self.code_generator = CodeGenerator()
        self.chat_history: List[Message] = []
        self.current_directory = os.getcwd()
    
    def process_request(self, user_input: str) -> str:
        """Process user coding request and return response"""
        # Add user message to history
        self.chat_history.append(Message("user", user_input))
        
        # Check if this is a coding request
        if not self._is_coding_request(user_input):
            response = self._handle_non_coding_request(user_input)
        else:
            # Plan tasks for coding request
            tasks = self.planner.plan_tasks(user_input)
            
            # Execute tasks
            results = []
            for task in tasks:
                result = self._execute_task(task)
                results.append(result)
            
            # Generate response
            response = self._generate_response(tasks, results, user_input)
        
        self.chat_history.append(Message("assistant", response))
        return response
    
    def _is_coding_request(self, user_input: str) -> bool:
        """Check if user input is a coding-related request"""
        coding_keywords = [
            'create', 'make', 'build', 'generate', 'write', 'code',
            'file', 'program', 'script', 'app', 'application',
            'project', 'game', 'calculator', 'tool'
        ]
        
        return any(keyword in user_input.lower() for keyword in coding_keywords)
    
    def _handle_non_coding_request(self, user_input: str) -> str:
        """Handle non-coding requests like file operations"""
        user_input_lower = user_input.lower()
        
        # File operations
        if 'list' in user_input_lower and ('file' in user_input_lower or 'directory' in user_input_lower):
            result = self.tools['filesystem'].execute('list_directory', path='.')
            return f"📁 Current directory contents:\n{result.output}"
        
        elif 'read' in user_input_lower and 'file' in user_input_lower:
            # Try to extract filename from input
            words = user_input_lower.split()
            filename = None
            for i, word in enumerate(words):
                if word == 'read' and i + 1 < len(words):
                    filename = words[i + 1]
                    break
            
            if filename:
                result = self.tools['filesystem'].execute('read_file', path=filename)
                if result.success:
                    return f"📄 Content of {filename}:\n{result.output}"
                else:
                    return f"❌ Could not read {filename}: {result.error}"
        
        elif 'delete' in user_input_lower and 'file' in user_input_lower:
            words = user_input_lower.split()
            filename = None
            for i, word in enumerate(words):
                if word == 'delete' and i + 1 < len(words):
                    filename = words[i + 1]
                    break
            
            if filename:
                result = self.tools['filesystem'].execute('delete_file', path=filename)
                return f"🗑️ {result.output}" if result.success else f"❌ {result.error}"
        
        return "🤔 I can help you with coding projects and file operations. Try something like:\n• 'Create a Python chess game'\n• 'Make a calculator in JavaScript'\n• 'Build a todo app'\n• 'List files in directory'\n• 'Read file example.py'"
    
    def _execute_task(self, task: Dict[str, Any]) -> ToolResult:
        """Execute a single task"""
        task_type = task['type']
        code_request = task['code_request']
        
        try:
            if task_type == 'create_environment':
                return self.tools['filesystem'].execute(
                    'create_project_env',
                    project_name=code_request.project_name,
                    language=code_request.language
                )
            
            elif task_type == 'generate_code':
                # Generate the code
                generated_code = self.code_generator.generate_code_from_description(code_request)
                
                # Create the main code file
                file_path = os.path.join(code_request.project_name, code_request.file_name)
                return self.tools['filesystem'].execute(
                    'create_file',
                    path=file_path,
                    content=generated_code
                )
            
            elif task_type == 'create_additional_files':
                return self._create_additional_files(code_request)
            
            else:
                return ToolResult(False, "", f"Unknown task type: {task_type}")
                
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _create_additional_files(self, code_request: CodeRequest) -> ToolResult:
        """Create additional files based on requested features"""
        created_files = []
        
        try:
            base_path = code_request.project_name
            
            # Create configuration files
            if 'database' in code_request.features and code_request.language == 'python':
                config_content = "# Database configuration\nDATABASE_URL = 'sqlite:///app.db'\n"
                config_path = os.path.join(base_path, 'config.py')
                self.tools['filesystem'].execute('create_file', path=config_path, content=config_content)
                created_files.append('config.py')
            
            # Create test files
            if code_request.language == 'python':
                test_content = f'''#!/usr/bin/env python3
"""
Test file for {code_request.project_name}
"""

import unittest
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Test{code_request.project_name.title().replace('_', '')}(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        self.assertTrue(True)  # Placeholder test
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        pass

if __name__ == '__main__':
    unittest.main()
'''
                test_path = os.path.join(base_path, 'tests', f'test_{code_request.file_name}')
                self.tools['filesystem'].execute('create_file', path=test_path, content=test_content)
                created_files.append(f'tests/test_{code_request.file_name}')
            
            # Create documentation
            if code_request.project_type in ['game', 'gui_app', 'web_app']:
                docs_content = f'''# {code_request.project_name.title()} Documentation

## Overview
{code_request.description}

## Features
{chr(10).join(f"- {feature}" for feature in code_request.features)}

## Installation

### Prerequisites
- {code_request.language.title()} installed on your system

### Setup
1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies (if any)
4. Run the application

## Usage

### Running the Application
```bash
{"python " + code_request.file_name if code_request.language == "python" else "node " + code_request.file_name if code_request.language == "javascript" else "Run " + code_request.file_name}
```

## Project Structure
```
{code_request.project_name}/
├── {code_request.file_name}     # Main application file
├── README.md                     # This file
├── requirements.txt             # Dependencies (Python)
└── tests/                       # Test files
```

## Contributing
1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License
This project is open source and available under the MIT License.
'''
                docs_path = os.path.join(base_path, 'DOCUMENTATION.md')
                self.tools['filesystem'].execute('create_file', path=docs_path, content=docs_content)
                created_files.append('DOCUMENTATION.md')
            
            if created_files:
                return ToolResult(True, f"Additional files created: {', '.join(created_files)}")
            else:
                return ToolResult(True, "No additional files needed")
                
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _generate_response(self, tasks: List[Dict], results: List[ToolResult], original_request: str) -> str:
        """Generate human-readable response"""
        success_count = sum(1 for result in results if result.success)
        total_count = len(results)
        
        if success_count == 0:
            return f"❌ Sorry, I couldn't complete your request. Please check the requirements and try again."
        
        # Get the code request from the first task
        code_request = tasks[0]['code_request'] if tasks else None
        
        response_parts = []
        
        if success_count == total_count:
            response_parts.append(f"🎉 Great! I've successfully created your {code_request.language} {code_request.project_type}!")
        else:
            response_parts.append(f"✅ I completed {success_count} out of {total_count} tasks:")
        
        # Add details about what was created
        if code_request:
            response_parts.append(f"\n📁 Project: {code_request.project_name}")
            response_parts.append(f"💻 Language: {code_request.language.title()}")
            response_parts.append(f"🚀 Type: {code_request.project_type.replace('_', ' ').title()}")
            
            if code_request.features:
                response_parts.append(f"⚡ Features: {', '.join(code_request.features)}")
        
        response_parts.append("\n📋 Tasks completed:")
        
        # Show task results
        for task, result in zip(tasks, results):
            status = "✅" if result.success else "❌"
            response_parts.append(f"{status} {task['description']}")
            if result.success and result.output:
                response_parts.append(f"   └─ {result.output}")
            elif not result.success and result.error:
                response_parts.append(f"   └─ Error: {result.error}")
        
        # Add usage instructions
        if success_count > 0 and code_request:
            response_parts.append(f"\n🏃 To run your project:")
            response_parts.append(f"   cd {code_request.project_name}")
            
            if code_request.language == 'python':
                response_parts.append(f"   python {code_request.file_name}")
            elif code_request.language == 'javascript':
                response_parts.append(f"   node {code_request.file_name}")
            elif code_request.language == 'java':
                response_parts.append(f"   javac {code_request.file_name} && java Main")
            elif code_request.language == 'html':
                response_parts.append(f"   Open index.html in your browser")
            else:
                response_parts.append(f"   Run {code_request.file_name}")
        
        return '\n'.join(response_parts)

# Example usage and testing
def main():
    """Main function for testing the agent"""
    agent = CodingAgent()
    
    print("🤖 General Coding Agent Started!")
    print("💡 I can create any type of code project in multiple languages!")
    print("🔤 Examples:")
    print("   • 'Create a Python chess game'")
    print("   • 'Make a JavaScript calculator'")
    print("   • 'Build a Java todo application'")
    print("   • 'Create a web scraper in Python'")
    print("   • 'Make a GUI app with tkinter'")
    print("\nType 'quit' to exit, 'help' for more examples")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🎯 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Happy coding!")
                break
            
            elif user_input.lower() in ['help', 'h']:
                print("\n📖 More Examples:")
                print("🎮 Games: 'Create a tic-tac-toe game in Python'")
                print("🧮 Tools: 'Make a password generator in JavaScript'")
                print("🌐 Web: 'Build a simple web page with HTML and CSS'")
                print("📱 Apps: 'Create a contact manager in Java'")
                print("🔧 Utils: 'Make a file organizer script in Python'")
                print("📊 Data: 'Create a CSV processor in Python'")
                continue
            
            elif user_input.lower() in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if user_input:
                print("\n🔄 Processing your request...")
                response = agent.process_request(user_input)
                print(f"\n🤖 Agent: {response}")
                print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Agent interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()