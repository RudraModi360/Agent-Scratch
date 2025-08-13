# Agent.py – General Purpose AI Agent (Groq-powered)

## Overview

**Agent.py** is a modular, extensible, general-purpose AI agent powered by Groq LLM. It can automate code generation, file management, research, data analysis, creative writing, system administration, and more. The agent is designed for developer productivity, research, and automation tasks, with a CLI interface and a robust tool system.

---

## Features

- **Groq LLM Integration**: Uses Groq for advanced reasoning, code generation, and planning.
- **Agentic Loop**: Iterative planning, execution, and reflection for complex tasks.
- **Extensible Tool System**: Easily add new tools for file, command, web, and data operations.
- **CLI Interface**: Interactive command-line experience for task automation.
- **Task Analysis & Planning**: Automatic breakdown of user requests into actionable steps.
- **File & Directory Management**: Create, read, write, search, and organize files.
- **Web Search & Research**: Real-time data gathering and web scraping.
- **Data Processing**: Analyze, transform, and summarize data (JSON, CSV, text).
- **Creative & Code Generation**: Generate code, stories, articles, and more.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Agent_Testing_Claude
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
**Main dependencies:**
- groq
- pydantic
- requests
- (others as needed)

### 3. Set Up Environment Variables
You **must** set your Groq API key:
```bash
export GROQ_API_KEY='your_groq_api_key_here'
```
Get your API key from: https://console.groq.com/keys

---

## Usage

### Command Line Interface (CLI)
Run the agent interactively:
```bash
python Agent.py
```

#### CLI Commands
- `help`      – Show help message
- `tools`     – List available tools
- `history`   – Show recent execution history
- `clear`     – Clear chat history
- `status`    – Show agent status
- `quit/exit` – Exit the agent

Or simply type your request (e.g. "Create a Python web scraper for news headlines").

### Programmatic Usage
You can import and use the agent in your own Python scripts:
```python
from Agent import GeneralGroqAgent
agent = GeneralGroqAgent(groq_api_key)
result = agent.process_request("Write a Python script to sort a CSV file")
print(result['final_response'])
```

---

## Architecture & Main Classes

- **GeneralGroqAgent**: Main agent class. Handles request processing, planning, execution, and history.
- **GroqLLMEngine**: Integrates Groq LLM for analysis, planning, and content generation.
- **BaseTool**: Abstract base for all tools. Extend this to add new capabilities.
- **FileSystemTool**: File and directory operations.
- **EditFileTool**: Exact text replacement in files.
- **CommandTool**: System command execution.
- **WebTool**: Web requests and scraping.
- **DataProcessingTool**: Data analysis and transformation.
- **AgentCLI**: Interactive command-line interface.

---

## Tool System & Extensibility

Tools are modular and easy to extend. To add a new tool:
1. Subclass `BaseTool` and implement required methods (`name`, `description`, `parameters`, `execute`).
2. Add your tool to the agent:
   ```python
   agent.add_tool(MyNewTool())
   ```
3. The agent will automatically use it for relevant tasks.

---

## Current Tools

The agent comes with the following built-in tools:

| Tool Name         | Description                                              |
|-------------------|---------------------------------------------------------|
| filesystem        | File and directory operations (create, read, write, etc) |
| edit_file         | Exact text replacement in files                          |
| command           | System command execution                                 |
| web               | Web requests and basic web scraping                      |
| data_processing   | Data analysis and transformation (JSON, CSV, text)       |

You can list available tools in the CLI by typing `tools`.

---

## Example Requests
- "Create a backup script for my documents"
- "Analyze this CSV file and show statistics"
- "Write a short story about space exploration"
- "List all Python files in the current directory"
- "Search the web for latest AI news"

---

## Troubleshooting
- **Missing GROQ_API_KEY**: Set your API key as described above.
- **Internet Connection**: Required for Groq LLM and web operations.
- **Dependencies**: Ensure all required Python packages are installed.
- **File Permissions**: Some operations may require appropriate file system permissions.

---

## Contribution Guidelines
- Fork the repo and create a feature branch.
- Add new tools by subclassing `BaseTool`.
- Write clear docstrings and comments.
- Submit pull requests with a description of your changes.

---

## License
MIT License (or specify your license here)

---

## Contact & Support
For issues, feature requests, or questions, open a GitHub issue or contact the maintainer.

---

## Quick Start
```bash
# 1. Clone and install
pip install -r requirements.txt
# 2. Set your Groq API key
export GROQ_API_KEY='your_groq_api_key_here'
# 3. Run the agent
python Agent.py
```

---

## FAQ
**Q: What tasks can Agent.py handle?**
A: Code generation, file management, research, data analysis, creative writing, automation, and more.

**Q: How do I add a new tool?**
A: Subclass `BaseTool` and add it to the agent with `add_tool()`.

**Q: Where do I get a Groq API key?**
A: https://console.groq.com/keys

**Q: Can I use this agent in my own scripts?**
A: Yes! See the Programmatic Usage section above.

---

Happy hacking! 🚀
