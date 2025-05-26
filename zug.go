package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// --- Autonomous Coding Agent ---
// A terminal-based agent that plans, writes, tests, and refines code
// similar to Claude Code or Plandex AI.

// AutonomousCodingAgent holds the state for our agent
type AutonomousCodingAgent struct {
	Client      *openai.Client
	ProjectDir  string
	Context     []openai.ChatCompletionMessage
}

// NewAutonomousCodingAgent creates a new agent
func NewAutonomousCodingAgent(apiKey string, projectDir string) (*AutonomousCodingAgent, error) {
	if projectDir == "" {
		projectDir = "project"
	}
	err := os.MkdirAll(projectDir, os.ModePerm)
	if err != nil {
		return nil, fmt.Errorf("error creating project directory: %w", err)
	}

	client := openai.NewClient(apiKey)
	return &AutonomousCodingAgent{
		Client:     client,
		ProjectDir: projectDir,
		Context:    []openai.ChatCompletionMessage{},
	}, nil
}

// askModel queries the OpenAI model
func (a *AutonomousCodingAgent) askModel(prompt string, temperature float32) (string, error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful coding assistant.",
		},
	}
	messages = append(messages, a.Context...)
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	// Define functions available to the model
	functions := []openai.FunctionDefinition{
		{
			Name:        "create_file",
			Description: "Create a new file with given content",
			Parameters: &openai.JSONSchemaDefinition{
				Type: openai.JSONSchemaTypeObject,
				Properties: map[string]*openai.JSONSchemaDefinition{
					"path":    {Type: openai.JSONSchemaTypeString},
					"content": {Type: openai.JSONSchemaTypeString},
				},
				Required: []string{"path", "content"},
			},
		},
		{
			Name:        "append_file",
			Description: "Append content to an existing file",
			Parameters: &openai.JSONSchemaDefinition{
				Type: openai.JSONSchemaTypeObject,
				Properties: map[string]*openai.JSONSchemaDefinition{
					"path":    {Type: openai.JSONSchemaTypeString},
					"content": {Type: openai.JSONSchemaTypeString},
				},
				Required: []string{"path", "content"},
			},
		},
	}

	resp, err := a.Client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:        openai.GPT4o, // Or your preferred model like gpt-4-0613
			Messages:     messages,
			Temperature:  temperature,
			Functions:    functions,
			FunctionCall: "auto",
		},
	)

	if err != nil {
		return "", fmt.Errorf("chat completion error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from API")
	}

	message := resp.Choices[0].Message
	a.Context = append(a.Context, openai.ChatCompletionMessage{
		Role:         message.Role,
		Content:      message.Content,
		FunctionCall: message.FunctionCall,
	})

	// Handle function calls
	if message.FunctionCall != nil {
		functionName := message.FunctionCall.Name
		var args map[string]interface{}
		err := json.Unmarshal([]byte(message.FunctionCall.Arguments), &args)
		if err != nil {
			return "", fmt.Errorf("error unmarshalling function arguments: %w", err)
		}

		path, pathOk := args["path"].(string)
		content, contentOk := args["content"].(string)

		if !pathOk || !contentOk {
			return "", fmt.Errorf("invalid arguments for function %s", functionName)
		}

		switch functionName {
		case "create_file":
			return a.createFile(path, content)
		case "append_file":
			return a.appendFile(path, content)
		default:
			return "", fmt.Errorf("unknown function call: %s", functionName)
		}
	}

	return message.Content, nil
}

// createFile creates a new file with the given content
func (a *AutonomousCodingAgent) createFile(path string, content string) (string, error) {
	fullPath := filepath.Join(a.ProjectDir, path)
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return "", fmt.Errorf("error creating directory %s: %w", dir, err)
	}

	err := os.WriteFile(fullPath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("error writing file %s: %w", fullPath, err)
	}
	return fmt.Sprintf("File created: %s", fullPath), nil
}

// appendFile appends content to an existing file
func (a *AutonomousCodingAgent) appendFile(path string, content string) (string, error) {
	fullPath := filepath.Join(a.ProjectDir, path)
	f, err := os.OpenFile(fullPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return "", fmt.Errorf("error opening file %s for appending: %w", fullPath, err)
	}
	defer f.Close()

	if _, err := f.WriteString(content); err != nil {
		return "", fmt.Errorf("error appending to file %s: %w", fullPath, err)
	}
	return fmt.Sprintf("Appended to: %s", fullPath), nil
}

// runShell executes a shell command in the project directory
func (a *AutonomousCodingAgent) runShell(cmdStr string) (string, error) {
	parts := strings.Fields(cmdStr)
	cmd := exec.Command(parts[0], parts[1:]...)
	cmd.Dir = a.ProjectDir

	var outb, errb bytes.Buffer
	cmd.Stdout = &outb
	cmd.Stderr = &errb

	err := cmd.Run()
	if err != nil {
		// Include stderr in the error message if the command failed
		return outb.String() + errb.String(), fmt.Errorf("error running command '%s': %w\nStderr: %s", cmdStr, err, errb.String())
	}
	return outb.String() + errb.String(), nil // Combine stdout and stderr
}

// feedbackLoop is the primary loop: ask model, execute, test, and evaluate
func (a *AutonomousCodingAgent) feedbackLoop(instruction string) {
	log.Println("[Agent] Starting task: ", instruction)
	currentInstruction := instruction

	for {
		output, err := a.askModel(currentInstruction, 0.2)
		if err != nil {
			log.Printf("[Agent] Error asking model: %v", err)
			// Potentially add retry logic or more sophisticated error handling here
			currentInstruction = fmt.Sprintf("The previous attempt to get a response failed with: %v. Please try again, or suggest a different approach to achieve: %s", err, instruction)
			if len(a.Context) > 5 { // Avoid overly long context on repeated errors
				log.Println("[Agent] Context getting long due to errors, attempting to reset instruction.")
				currentInstruction = instruction
				a.Context = []openai.ChatCompletionMessage{} // Reset context carefully
			}
			continue
		}
		log.Println(output)

		testsDir := filepath.Join(a.ProjectDir, "tests")
		if _, err := os.Stat(testsDir); !os.IsNotExist(err) {
			log.Println("[Agent] Running tests...")
			// Note: This assumes pytest is installed and in PATH.
			// Go's testing is usually done via `go test`.
			// If you want to run Python tests, you'd call pytest as an external command.
			// If you want to run Go tests, the command would be `go test ./...`
			// For this port, we'll keep the pytest assumption.
			testOutput, err := a.runShell("pytest --maxfail=1 --disable-warnings -q")
			log.Println(testOutput)
			if err != nil {
				log.Printf("[Agent] Error running tests: %v", err)
				// The error from runShell already contains stderr
				currentInstruction = fmt.Sprintf("Tests command failed to execute. Here is the command output:\n%s\nPlease check the test setup or fix the code based on this. Original task: %s", testOutput, instruction)
				continue
			}

			if !strings.Contains(strings.ToLower(testOutput), "failed") && !strings.Contains(strings.ToLower(testOutput), "error") {
				log.Println("[Agent] All tests passed 🎉")
				break
			} else {
				log.Println("[Agent] Tests failed, asking for fixes...")
				currentInstruction = fmt.Sprintf("Tests failed. Here is the output:\n%s\nPlease fix the errors. Original task: %s", testOutput, instruction)
			}
		} else {
			log.Println("[Agent] No tests found. Task complete.")
			break
		}
	}
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Error: Please set your OPENAI_API_KEY environment variable.")
	}

	if len(os.Args) < 2 {
		log.Fatal("Usage: go run autonomous_agent.go \"Describe your coding task\"")
	}

	task := os.Args[1]

	agent, err := NewAutonomousCodingAgent(apiKey, "project_go") // Changed default project dir
	if err != nil {
		log.Fatalf("Error creating agent: %v", err)
	}

	agent.feedbackLoop(task)
}

// Instructions:
// 1. Install Go: https://go.dev/doc/install
// 2. Install OpenAI Go library: go get github.com/sashabaranov/go-openai
// 3. Set API Key: export OPENAI_API_KEY="your_key"
// 4. Run: go run your_file_name.go "Build a simple Go web server with a health check endpoint and unit tests"
//    (Assuming this file is named your_file_name.go)
// 5. If using pytest for tests, ensure it's installed and the project structure allows it.
//    Alternatively, adapt the test execution to use Go's native testing (`go test`).
