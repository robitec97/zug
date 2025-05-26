package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ────────────────────────────────────────────────
//  AutonomousCodingAgent — a multi-step GPT-4o driven
//  coding assistant that can iteratively read/modify the
//  local code-base, run shell commands & execute tests.
// ────────────────────────────────────────────────

type AutonomousCodingAgent struct {
	client         *openai.Client
	projectDir     string
	ctx            []openai.ChatCompletionMessage
	maxCtxMessages int // sliding-window for conversation history
}

func NewAgent(apiKey, projectDir string) *AutonomousCodingAgent {
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		log.Fatalf("cannot create project dir %s: %v", projectDir, err)
	}
	return &AutonomousCodingAgent{
		client:         openai.NewClient(apiKey),
		projectDir:     projectDir,
		maxCtxMessages: 40, // keep the last N messages to stay within budget
	}
}

/*──────────────────────────────
  Utility helpers
  ─────────────────────────────*/

// absPath ensures rel cannot traverse outside the sandbox.
func (a *AutonomousCodingAgent) absPath(rel string) (string, error) {
	clean := filepath.Clean(rel)
	// Prevent paths that are absolute or try to go "up" from the root.
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, ".."+string(os.PathSeparator)) || clean == ".." {
		return "", fmt.Errorf("invalid path %q (must be relative and stay within project dir)", rel)
	}
	full := filepath.Join(a.projectDir, clean)
	// Ensure the fully resolved path is truly within the projectDir.
	// Add path separator to projectDir to avoid partial matches (e.g. /foo/bar vs /foo/barbaz)
	cleanProjectDir := filepath.Clean(a.projectDir)
	if !strings.HasPrefix(full, cleanProjectDir+string(os.PathSeparator)) && full != cleanProjectDir {
		return "", fmt.Errorf("invalid path %q (escapes project dir)", rel)
	}
	return full, nil
}

/*──────────────────────────────
  File operations (tools)
  ─────────────────────────────*/

func (a *AutonomousCodingAgent) createFile(path, content string) (string, error) {
	full, err := a.absPath(path)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return "", fmt.Errorf("failed to create directory for %s: %w", path, err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("failed to write file %s: %w", path, err)
	}
	return fmt.Sprintf("file %s created", path), nil
}

func (a *AutonomousCodingAgent) appendFile(path, content string) (string, error) {
	full, err := a.absPath(path)
	if err != nil {
		return "", err
	}
	// Ensure directory exists before trying to open/create the file
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return "", fmt.Errorf("failed to create directory for %s: %w", path, err)
	}
	f, err := os.OpenFile(full, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", fmt.Errorf("failed to open/append to file %s: %w", path, err)
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		return "", fmt.Errorf("failed to write content to %s: %w", path, err)
	}
	return fmt.Sprintf("content appended to %s", path), nil
}

func (a *AutonomousCodingAgent) updateFile(path, find, replace string) (string, error) {
	full, err := a.absPath(path)
	if err != nil {
		return "", err
	}
	raw, err := os.ReadFile(full)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s for update: %w", path, err)
	}
	src := string(raw)
	var dst string
	// Attempt to compile the 'find' string as a regular expression.
	// If 'find' is not a valid regex, it will fall back to plain string replacement.
	if re, errRe := regexp.Compile(find); errRe == nil {
		dst = re.ReplaceAllString(src, replace)
	} else {
		log.Printf("[agent] Info: 'find' string \"%s\" is not a valid regex (%v). Performing plain text replacement for updateFile on %s.", find, errRe, path)
		dst = strings.ReplaceAll(src, find, replace)
	}

	if dst == src {
		return fmt.Sprintf("nothing replaced in %s (content was identical or find pattern did not match)", path), nil
	}
	if err := os.WriteFile(full, []byte(dst), 0o644); err != nil {
		return "", fmt.Errorf("failed to write updated content to %s: %w", path, err)
	}
	return fmt.Sprintf("updated %s", path), nil
}

func (a *AutonomousCodingAgent) readFile(path string) (string, error) {
	full, err := a.absPath(path)
	if err != nil {
		return "", err
	}
	raw, err := os.ReadFile(full)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", path, err)
	}
	return string(raw), nil
}

func (a *AutonomousCodingAgent) listFiles() (string, error) {
	var list []string
	projectRoot := filepath.Clean(a.projectDir)
	err := filepath.WalkDir(projectRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			// Log permission errors but try to continue if possible
			log.Printf("Warning: error accessing %s: %v. Skipping.", p, err)
			if errors.Is(err, fs.ErrPermission) {
				if d != nil && d.IsDir() {
					return fs.SkipDir // Skip this directory if permission denied
				}
				return nil // Skip this file
			}
			return err // Propagate other critical errors
		}
		if d.IsDir() {
			// Optionally skip common VCS or project-specific directories
			// e.g. if d.Name() == ".git" || d.Name() == ".idea" { return fs.SkipDir }
			return nil
		}
		rel, errRel := filepath.Rel(projectRoot, p)
		if errRel != nil {
			log.Printf("Warning: could not make path relative %s: %v", p, errRel)
			return errRel // Should not happen if p starts with a.projectDir
		}
		list = append(list, rel)
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("error listing files in %s: %w", projectRoot, err)
	}
	if len(list) == 0 {
		return "No files found in the project.", nil
	}
	return strings.Join(list, "\n"), nil
}

/*──────────────────────────────
  Shell runner (tool)
  ─────────────────────────────*/

func (a *AutonomousCodingAgent) runShell(cmd string) (string, error) {
	// For security, consider disallowing certain commands or patterns if this agent
	// could be exposed to untrusted input for the 'cmd' string.
	// For now, it executes what it's told within its projectDir.
	log.Printf("[agent] executing shell command: %s in %s\n", cmd, a.projectDir)
	c := exec.Command("bash", "-c", cmd)
	c.Dir = a.projectDir
	out, err := c.CombinedOutput() // Captures both stdout and stderr

	outputStr := strings.TrimSpace(string(out))

	if err != nil {
		// Return both output and error so the model can diagnose.
		// This is a specific design choice for this agent.
		log.Printf("[agent] shell command error: %v, output: %s\n", err, outputStr)
		return fmt.Sprintf("Output:\n%s\nERROR: %s", outputStr, err.Error()), nil
	}
	log.Printf("[agent] shell command output: %s\n", outputStr)
	return outputStr, nil
}

/*──────────────────────────────
  OpenAI interaction helpers
  ─────────────────────────────*/

// toolParams builds a minimal JSON schema {string:string, ...}.
func toolParams(keys ...string) map[string]interface{} {
	props := map[string]interface{}{}
	for _, k := range keys {
		props[k] = map[string]string{"type": "string"}
	}
	m := map[string]interface{}{
		"type":       "object",
		"properties": props,
	}
	// "required" should only be set if there are keys.
	if len(keys) > 0 {
		m["required"] = keys
	}
	return m
}

// toolDefs defines the tools available to the OpenAI model.
func (a *AutonomousCodingAgent) toolDefs() []openai.Tool {
	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "create_file",
				Description: "Create a new file with given content. Path should be relative to project root. Ensures parent directories exist.",
				Parameters:  toolParams("path", "content"),
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "append_file",
				Description: "Append content to an existing file. Creates the file if it doesn't exist. Path should be relative to project root.",
				Parameters:  toolParams("path", "content"),
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "update_file",
				Description: "Search (regex or plain text) & replace text in an existing file. 'find' can be a regex. Path should be relative to project root.",
				Parameters:  toolParams("path", "find", "replace"),
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "read_file",
				Description: "Read the contents of an existing file. Path should be relative to project root.",
				Parameters:  toolParams("path"),
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "list_files",
				Description: "List all files in the project, relative to project root. Returns 'No files found...' if empty.",
				Parameters:  toolParams(), // No parameters for list_files
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "run_shell",
				Description: "Execute a shell command (bash) in the project dir and return its combined stdout/stderr. Errors are included in the output.",
				Parameters:  toolParams("command"),
			},
		},
	}
}

// systemPrompt defines the initial system message for the AI.
func systemPrompt() openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleSystem,
		Content: `You are AutonomousCoder, a senior software engineer. Work step-by-step: decide which file to read or modify, or which shell command to run, using the provided tools. File paths should always be relative to the project root. Iterate until the tests pass or the goal is reached. Respond concisely. If a tool fails, analyze the error and try to fix the issue in your next step. If a shell command produces an error, that error will be part of its output. If a file operation results in 'nothing changed', consider if the 'find' pattern was correct or if the file already has the desired content. Be precise with file paths. Always use 'list_files' if unsure about file existence or names before attempting to read or write.`,
	}
}

// chat handles an entire cycle of user prompt → potential tool calls → assistant reply.
func (a *AutonomousCodingAgent) chat(userPrompt string, temperature float32) (string, error) {
	// Add current user prompt to the agent's context
	a.ctx = append(a.ctx, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: userPrompt})

	// Maintain sliding window for a.ctx before making any API call
	if len(a.ctx) > a.maxCtxMessages {
		cutoff := len(a.ctx) - a.maxCtxMessages
		// Keep the system prompt (if we were to always prepend it here) + the most recent messages
		// However, system prompt is added separately below, so just trim a.ctx
		a.ctx = a.ctx[cutoff:]
		log.Printf("[agent] Context trimmed to %d messages.\n", len(a.ctx))
	}

	// Prepare messages for the current API call, including the system prompt
	messagesForAPI := append([]openai.ChatCompletionMessage{systemPrompt()}, a.ctx...)

	// Loop for potential multiple tool calls within a single user turn
	for step := 0; step < 10; step++ { // Safety: max 10 tool hops per user turn
		log.Printf("[agent] Chat step %d. Sending %d messages to API (incl. system prompt).\n", step+1, len(messagesForAPI))

		req := openai.ChatCompletionRequest{
			Model:       openai.GPT4o, // Or other capable model like gpt-4-turbo
			Messages:    messagesForAPI,
			Temperature: temperature,
			Tools:       a.toolDefs(),
			ToolChoice:  "auto", // Use string "auto"
			MaxTokens:   1500,   // Increased for potentially complex responses or tool args
		}

		resp, err := a.client.CreateChatCompletion(context.Background(), req)
		if err != nil {
			// If API call fails, the last user message and any subsequent optimistic additions to a.ctx might need rollback
			// For now, just return error. The caller (feedbackLoop) might retry or fail.
			return "", fmt.Errorf("CreateChatCompletion failed on step %d: %w", step+1, err)
		}

		if len(resp.Choices) == 0 {
			return "", errors.New("received an empty Choices array from OpenAI")
		}
		msg := resp.Choices[0].Message

		// Add assistant's response (which might be a content response or a tool call request) to agent's context
		a.ctx = append(a.ctx, msg)
		// Also add it to messagesForAPI for the *next* iteration of this tool-use loop, if any
		messagesForAPI = append(messagesForAPI, msg)


		// If no tool calls, assistant provided a direct content response. This turn is over.
		if len(msg.ToolCalls) == 0 {
			if msg.Content == "" {
				log.Println("[agent] Warning: Assistant response has no tool calls and no content.")
				return "", errors.New("assistant provided no content and no tool calls")
			}
			log.Printf("[agent] Assistant response (no tool call): %s\n", msg.Content)
			return msg.Content, nil
		}

		// If there are tool calls, process them.
		log.Printf("[agent] Assistant requests %d tool call(s).\n", len(msg.ToolCalls))
		for _, toolCall := range msg.ToolCalls {
			if toolCall.Type == openai.ToolTypeFunction {
				toolName := toolCall.Function.Name
				toolArgs := toolCall.Function.Arguments
				log.Printf("[agent] Tool call requested: %s(%s)\n", toolName, toolArgs)

				toolResult, toolErr := a.execTool(toolName, toolArgs)
				if toolErr != nil {
					log.Printf("[agent] Tool %s execution error: %v\n", toolName, toolErr)
					// Format error message for the LLM to understand
					toolResult = fmt.Sprintf("TOOL_EXECUTION_ERROR for %s: %s", toolName, toolErr.Error())
				} else {
					log.Printf("[agent] Tool %s result: %s\n", toolName, toolResult)
				}

				toolResponseMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					ToolCallID: toolCall.ID, // Crucial: Link the result to the specific call
					Name:       toolName,
					Content:    toolResult,
				}
				// Add tool response to agent's context
				a.ctx = append(a.ctx, toolResponseMessage)
				// Also add it to messagesForAPI for the next iteration of this tool-use loop
				messagesForAPI = append(messagesForAPI, toolResponseMessage)
			} else {
				log.Printf("[agent] Warning: Received unhandled tool type: %s\n", toolCall.Type)
				// Add a placeholder message to context if necessary, or handle appropriately
				errorMsg := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					ToolCallID: toolCall.ID,
					Name:       toolCall.Function.Name, // Assuming it might have a name
					Content:    fmt.Sprintf("Error: Tool type '%s' is not supported by the agent.", toolCall.Type),
				}
				a.ctx = append(a.ctx, errorMsg)
				messagesForAPI = append(messagesForAPI, errorMsg)
			}
		}
		// After processing all tool calls for this step, trim context again for the next API call in this loop
		if len(a.ctx) > a.maxCtxMessages {
			cutoff := len(a.ctx) - a.maxCtxMessages
			a.ctx = a.ctx[cutoff:]
			log.Printf("[agent] Context trimmed to %d messages during tool loop.\n", len(a.ctx))
			// Rebuild messagesForAPI based on the newly trimmed a.ctx for the next step
			messagesForAPI = append([]openai.ChatCompletionMessage{systemPrompt()}, a.ctx...)
		}
		// Continue the loop to let the model react to the tool result(s).
	}
	log.Println("[agent] Error: Exceeded maximum tool invocations for this turn.")
	return "", errors.New("too many tool invocations without a final answer")
}


// execTool deserialises args and dispatches to the matching Go helper.
func (a *AutonomousCodingAgent) execTool(name, jsonArgs string) (string, error) {
	log.Printf("[agent] execTool: %s, Args: %s\n", name, jsonArgs)
	switch name {
	case "create_file", "append_file":
		var p struct {
			Path    string `json:"path"`
			Content string `json:"content"` // Content can be empty for create_file (empty file)
		}
		if err := json.Unmarshal([]byte(jsonArgs), &p); err != nil {
			return "", fmt.Errorf("invalid JSON arguments for %s: %w. Raw args: %s", name, err, jsonArgs)
		}
		if strings.TrimSpace(p.Path) == "" {
			return "", fmt.Errorf("argument 'path' for %s cannot be empty. Raw args: %s", name, jsonArgs)
		}
		if name == "create_file" {
			return a.createFile(p.Path, p.Content)
		}
		return a.appendFile(p.Path, p.Content)

	case "update_file":
		var p struct {
			Path    string `json:"path"`
			Find    string `json:"find"`    // Find can be empty, meaning replace entire content if replace is not empty
			Replace string `json:"replace"` // Replace can be empty, meaning delete found content
		}
		if err := json.Unmarshal([]byte(jsonArgs), &p); err != nil {
			return "", fmt.Errorf("invalid JSON arguments for %s: %w. Raw args: %s", name, err, jsonArgs)
		}
		if strings.TrimSpace(p.Path) == "" {
			return "", fmt.Errorf("argument 'path' for %s cannot be empty. Raw args: %s", name, jsonArgs)
		}
		return a.updateFile(p.Path, p.Find, p.Replace)

	case "read_file":
		var p struct{ Path string `json:"path"` }
		if err := json.Unmarshal([]byte(jsonArgs), &p); err != nil {
			return "", fmt.Errorf("invalid JSON arguments for read_file: %w. Raw args: %s", err, jsonArgs)
		}
		if strings.TrimSpace(p.Path) == "" {
			return "", fmt.Errorf("argument 'path' for read_file cannot be empty. Raw args: %s", jsonArgs)
		}
		return a.readFile(p.Path)

	case "list_files":
		// No arguments expected, jsonArgs might be "{}" or empty.
		// Validate that jsonArgs is indeed empty or an empty object if strict.
		var p map[string]interface{}
		if err := json.Unmarshal([]byte(jsonArgs), &p); err != nil {
			return "", fmt.Errorf("invalid JSON arguments for list_files (expected empty or {}): %w. Raw args: %s", err, jsonArgs)
		}
		if len(p) != 0 {
			return "", fmt.Errorf("list_files expects no arguments, but received some. Raw args: %s", jsonArgs)
		}
		return a.listFiles()

	case "run_shell":
		var p struct{ Command string `json:"command"` }
		if err := json.Unmarshal([]byte(jsonArgs), &p); err != nil {
			return "", fmt.Errorf("invalid JSON arguments for run_shell: %w. Raw args: %s", err, jsonArgs)
		}
		if strings.TrimSpace(p.Command) == "" {
			return "", fmt.Errorf("argument 'command' for run_shell cannot be empty. Raw args: %s", jsonArgs)
		}
		return a.runShell(p.Command)

	default:
		return "", fmt.Errorf("unknown tool %q requested by LLM", name)
	}
}

/*──────────────────────────────
  Feedback-driven loop
  ─────────────────────────────*/

func (a *AutonomousCodingAgent) feedbackLoop(initialTask string) {
	log.Printf("[agent] 🏁 Starting main task: %s\n", initialTask)
	currentTaskInstruction := initialTask

	// Overall loop for iterative refinement based on tests or other feedback
	for turn := 0; turn < 10; turn++ { // Max 10 overall turns for the task
		log.Printf("[agent] >>> Feedback Loop Turn %d/%d. Current instruction: %s\n", turn+1, 10, currentTaskInstruction)

		// The 'chat' function itself has an inner loop for tool usage.
		// This outer loop is for broader feedback, like test results.
		assistantReply, err := a.chat(currentTaskInstruction, 0.1) // Lower temp for more deterministic tool use
		if err != nil {
			// If chat fails (e.g. too many tool steps, API error), decide how to proceed.
			// Maybe retry once, or modify the task, or give up.
			log.Printf("❌ Model interaction (chat function) failed on turn %d: %v. Aborting this task.", turn+1, err)
			// Potentially add the error to context for a final attempt, or just exit.
			// For now, we exit the feedback loop.
			return
		}
		fmt.Printf("🤖 Assistant's Plan/Summary:\n%s\n\n", assistantReply)

		// Check for tests after the assistant believes it has made progress or completed a step.
		testsDir := filepath.Join(a.projectDir, "tests")
		if info, statErr := os.Stat(testsDir); statErr == nil && info.IsDir() {
			log.Println("[agent] Running tests in 'tests/' directory...")
			// Standardize test command or make it configurable.
			// Assuming pytest for Python projects.
			testCommand := "pytest -q --maxfail=1 --disable-warnings tests/"
			testOutput, _ := a.runShell(testCommand) // runShell already handles errors by including them in output

			fmt.Printf("🐍 Test Execution Output:\n%s\n\n", testOutput)

			// Basic check for "failed" or "error" in output. Could be more sophisticated.
			// Also check for "passed" to confirm success.
			// Note: Pytest exit codes are more reliable if runShell could return them.
			// 0: All tests passed
			// 1: Tests were collected and run but some failed/errored
			// 2: Test execution was interrupted by the user
			// 3: Internal error in pytest
			// 4: pytest command line usage error
			// 5: No tests were collected
			// For now, relying on string parsing of output.
			if !strings.Contains(strings.ToLower(testOutput), "fail") &&
				!strings.Contains(strings.ToLower(testOutput), "error") &&
				(strings.Contains(strings.ToLower(testOutput), "pass") || strings.Contains(strings.ToLower(testOutput), "no tests ran")) {

				if strings.Contains(strings.ToLower(testOutput), "no tests ran") && !strings.Contains(strings.ToLower(testOutput), "collected 0 items") {
					log.Println("[agent] 🤔 Tests reported 'no tests ran' but it wasn't 'collected 0 items'. This might indicate a test discovery issue. Assuming success for now but please verify.")
				}
				log.Println("[agent] ✅ All tests passed (or no tests failed/errored). Task considered complete.")
				return // Successfully exit feedbackLoop
			}
			log.Println("[agent] 🔬 Tests failed or encountered errors.")
			currentTaskInstruction = fmt.Sprintf("The previous operations led to test failures. Please analyze the following test output and fix the code. Test output:\n%s", testOutput)
			time.Sleep(1 * time.Second) // Brief pause before formulating the next request to the LLM
		} else {
			log.Printf("[agent] 🎉 Task processing by assistant is complete. No 'tests' directory found at '%s' or it's not a directory. Manual verification recommended.\n", testsDir)
			return // Successfully exit feedbackLoop, assuming task is done if no tests.
		}
	}
	log.Println("[agent] ⚠️ Reached maximum turns in feedback loop. Task may not be fully complete or tests might still be failing.")
}

/*──────────────────────────────
  main
  ─────────────────────────────*/

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile) // Add file/line number to logs for easier debugging

	if len(os.Args) < 2 || strings.TrimSpace(os.Args[1]) == "" {
		fmt.Printf("Usage: %s \"<describe your coding task>\"\n", os.Args[0])
		fmt.Println("Example: go run . \"Create a Python script in main.py that prints 'Hello, World!' and add a test for it in tests/test_main.py\"")
		os.Exit(1)
	}
	initialTask := os.Args[1]

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("FATAL: OPENAI_API_KEY environment variable is not set.")
	}

	// Define a project directory. This will be created if it doesn't exist.
	// Using a subdirectory helps keep generated files organized.
	projectDirName := "ai_coder_project"
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("FATAL: Could not get current working directory: %v", err)
	}
	projectFullPath := filepath.Join(wd, projectDirName)

	log.Printf("[agent] Project directory will be: %s\n", projectFullPath)
	log.Printf("[agent] Initial task from command line: %s\n", initialTask)

	agent := NewAgent(apiKey, projectFullPath)

	// Initialize context with a simple message if needed, or let chat() handle the first user message.
	// agent.ctx = []openai.ChatCompletionMessage{} // Start with empty context for chat() to populate

	agent.feedbackLoop(initialTask)

	log.Println("[agent] 🏁 Autonomous Coding Agent finished.")
}
