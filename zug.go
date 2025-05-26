package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// AutonomousCodingAgent is our main struct
type AutonomousCodingAgent struct {
	client     *openai.Client
	projectDir string
	context    []openai.ChatCompletionMessage
}

// NewAgent constructs the agent, ensuring the project directory exists.
func NewAgent(apiKey, projectDir string) *AutonomousCodingAgent {
	if err := os.MkdirAll(projectDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating project directory: %v\n", err)
		os.Exit(1)
	}
	client := openai.NewClient(apiKey)
	return &AutonomousCodingAgent{
		client:     client,
		projectDir: projectDir,
		context:    []openai.ChatCompletionMessage{},
	}
}

// askModel sends a chat-completion request, handles function calls, and returns the model's reply.
func (a *AutonomousCodingAgent) askModel(prompt string, temperature float32) (string, error) {
	// prepend system prompt + history
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "You are a helpful coding assistant."},
	}
	messages = append(messages, a.context...)
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	// define our two function specs
	functions := []openai.FunctionDefinition{
		{
			Name:        "create_file",
			Description: "Create a new file with given content",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path":    map[string]string{"type": "string"},
					"content": map[string]string{"type": "string"},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			Name:        "append_file",
			Description: "Append content to an existing file",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path":    map[string]string{"type": "string"},
					"content": map[string]string{"type": "string"},
				},
				"required": []string{"path", "content"},
			},
		},
	}

	resp, err := a.client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:            openai.GPT4,
			Messages:         messages,
			Temperature:      temperature,
			Functions:        functions,
			FunctionCall:     "auto",
			MaxTokens:        1500,
			TopP:             1,
			FrequencyPenalty: 0,
			PresencePenalty:  0,
		},
	)
	if err != nil {
		return "", err
	}

	choice := resp.Choices[0].Message
	// save to history
	a.context = append(a.context, choice)

	// if a function call happened
	if choice.FunctionCall != nil {
		name := choice.FunctionCall.Name
		var args struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(choice.FunctionCall.Arguments), &args); err != nil {
			return "", err
		}

		switch name {
		case "create_file":
			return a.createFile(args.Path, args.Content)
		case "append_file":
			return a.appendFile(args.Path, args.Content)
		default:
			return "", fmt.Errorf("unknown function: %s", name)
		}
	}

	return choice.Content, nil
}

// createFile writes a new file under projectDir
func (a *AutonomousCodingAgent) createFile(relPath, content string) (string, error) {
	fullPath := filepath.Join(a.projectDir, relPath)
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(fullPath, []byte(content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("File created: %s", fullPath), nil
}

// appendFile appends content to an existing file
func (a *AutonomousCodingAgent) appendFile(relPath, content string) (string, error) {
	fullPath := filepath.Join(a.projectDir, relPath)
	f, err := os.OpenFile(fullPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		return "", err
	}
	return fmt.Sprintf("Appended to: %s", fullPath), nil
}

// runShell executes a shell command in projectDir and returns its combined output
func (a *AutonomousCodingAgent) runShell(cmd string) (string, error) {
	parts := strings.Fields(cmd)
	c := exec.Command(parts[0], parts[1:]...)
	c.Dir = a.projectDir
	out, err := c.CombinedOutput()
	return string(out), err
}

// feedbackLoop is the main driver: it asks for code, runs tests, loops until success
func (a *AutonomousCodingAgent) feedbackLoop(instruction string) {
	fmt.Println("[Agent] Starting task:", instruction)
	for {
		output, err := a.askModel(instruction, 0.2)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error asking model: %v\n", err)
			return
		}
		fmt.Println(output)

		testsDir := filepath.Join(a.projectDir, "tests")
		if info, err := os.Stat(testsDir); err == nil && info.IsDir() {
			fmt.Println("[Agent] Running tests...")
			testOut, _ := a.runShell("pytest --maxfail=1 --disable-warnings -q")
			fmt.Println(testOut)
			if !strings.Contains(strings.ToLower(testOut), "failed") {
				fmt.Println("[Agent] All tests passed 🎉")
				break
			}
			fmt.Println("[Agent] Tests failed, asking for fixes…")
			instruction = fmt.Sprintf("Tests failed. Here is the output:\n%s\nPlease fix the errors.", testOut)
			// slight pause so logs don't jumble
			time.Sleep(1 * time.Second)
		} else {
			fmt.Println("[Agent] No tests found. Task complete.")
			break
		}
	}
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "Error: Please set your OPENAI_API_KEY environment variable.")
		os.Exit(1)
	}
	agent := NewAgent(apiKey, "project")

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: go run main.go \"Describe your coding task\"")
		os.Exit(1)
	}
	task := os.Args[1]
	agent.feedbackLoop(task)
}
