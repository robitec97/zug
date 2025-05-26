# Zug – Terminal-based Autonomous Coding Agent

**Zug** is a lightweight, fast, and minimal terminal-based autonomous coding agent written in pure Go. It leverages an LLM of your choice to plan, write, test, and iterate on code from natural language instructions. Inspired by tools like Claude Code and Plandex AI, Zug is built to be simple and performant, a quality that is becoming rare today.

---

## ✨ Features

* 🧠 **Autonomous Task Execution**: Reads your coding prompt, plans, creates files, runs code, tests output, and iterates automatically.
* ⚡ **Lightweight & Fast**: Built in pure Go with minimal external dependencies for speed and portability.
* 📂 **File Manipulation**: Supports creating and appending to files through AI-driven commands.
* 🔧 **Minimal & Simple**: A single Go file with clear logic unlike bloated frameworks.

---

## 🚀 Usage

### Prerequisites

* [Go](https://go.dev/doc/install)
* OpenAI API Key

### Setup

```bash
git clone https://github.com/yourusername/zug.git
cd zug
go get github.com/sashabaranov/go-openai
```

### Build & Run

```bash
export OPENAI_API_KEY="your_openai_api_key"
go build zug.go 
./zug "Build a simple Go web server with a health check endpoint and unit tests"
```

The agent will:

* Generate code based on your instruction
* Save files in the `project_go/` directory
* Run tests if found
* Iterate on failures until the goal is reached
---

## 🧠 Why Zug?

### ✅ **Lightweight & Fast**

No bloated Python chains or orchestration layers. Zug is pure Go: fast to build, fast to run, easy to understand.

### ✅ **Minimal, Understandable Design**

All logic is contained in a single, readable Go file. Great for hacking, auditing, and extending.

### 🔮 **Coming Soon: Multi-Provider AI Support**

Support for Claude, Gemini, Ollama (self-hosted), and other providers is planned abstracting the model layer for maximum flexibility.

---

## 📌 Example Prompt

```bash
./zug "Create a CLI tool in Go that converts temperatures between Celsius and Fahrenheit and includes tests"
```

---

## 💡 Future Ideas

* Multiple model backends (Claude, Gemini, Ollama)
* Web UI or TUI frontend
* Persistent memory and task history
* Advanced file operations and refactoring commands

---

## 📄 License

MIT – free to use, share, and build upon.

---

Made with ❤️ by \[robitec97]
