package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"
)

var installCmd = &cobra.Command{
	Use:   "install",
	Short: "Install the Codegate extension and setup the configuration",
	RunE:  run,
}

func init() {
	installCmd.Flags().BoolP("dry-run", "r", false, "Only dry run the installation")
}

func main() {
	if err := installCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// Config represents the Continue configuration structure
type Config struct {
	Models               []Model    `json:"models"`
	ModelRoles           ModelRoles `json:"modelRoles"`
	TabAutocompleteModel *Model     `json:"tabAutocompleteModel,omitempty"`
}

type Model struct {
	Title    string `json:"title"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
	APIKey   string `json:"apiKey"`
	APIBase  string `json:"apiBase"`
}

type ModelRoles struct {
	Default string `json:"default"`
}

func run(cmd *cobra.Command, args []string) error {
	// Check required tools
	if err := checkPrerequisites(); err != nil {
		return fmt.Errorf("could not verify prerequisites: %w", err)
	}

	// We set up Docker first, then configuration, and finally the extension
	// so that a partial failure does not change the user-visible state.
	if err := setupDocker(); err != nil {
		return fmt.Errorf("failed to setup Docker: %w", err)
	}

	// Setup configuration
	if err := setupConfig(); err != nil {
		return fmt.Errorf("failed to setup configuration: %w", err)
	}

	// Install VS Code extension
	if err := installVSCodeExtension(); err != nil {
		return fmt.Errorf("failed to install VS Code extension: %w", err)
	}

	return nil
}

func checkPrerequisites() error {
	fmt.Println("Checking if Docker is installed...")

	// Check Docker installation
	if _, err := exec.LookPath("docker"); err != nil {
		return fmt.Errorf("Docker is not installed: %w", err)
	}

	// Make sure Docker is _actually_ running, not just installed
	if err := exec.Command("docker", "info").Run(); err != nil {
		return fmt.Errorf("Docker is not running: %w", err)
	}

	// Check Docker Compose Installation
	if err := exec.Command("docker", "compose", "version").Run(); err != nil {
		if _, err := exec.LookPath("docker-compose"); err != nil {
			return fmt.Errorf("Docker Compose is not installed: %w", err)
		}
	}

	return nil
}

func installVSCodeExtension() error {
	var extensions = []string{
		"continue.continue",
	}

	fmt.Println("Installing Continue extension...")

	for _, extension := range extensions {
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "windows":
			cmd = exec.Command("code.cmd", "--install-extension", "continue.continue", "--force")
		default:
			cmd = exec.Command("code", "--install-extension", "continue.continue", "--force")
		}

		output, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to install extension %s:\n %s: %w", extension, string(output), err)
		}
	}

	fmt.Println("Continue extension installed successfully!")
	return nil
}

func setupConfig() error {
	fmt.Println("Setting up config to use stacklok-hosted model...")

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	configDir := filepath.Join(homeDir, ".continue")
	configFile := filepath.Join(configDir, "config.json")

	// Create config directory if it doesn't exist
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Create default config if it doesn't exist
	config := Config{}

	// Read existing config if it exists
	if _, err := os.Stat(configFile); err == nil {
		// Backup existing config
		if err := copyFile(configFile, configFile+".bak"); err != nil {
			return fmt.Errorf("failed to backup config: %w", err)
		}

		existingConfig, err := os.ReadFile(configFile)
		if err != nil {
			return fmt.Errorf("failed to read config: %w", err)
		}

		if err := json.Unmarshal(existingConfig, &config); err != nil {
			return fmt.Errorf("failed to parse config: %w", err)
		}

	}
	config.ModelRoles.Default = "stacklok-hosted"

	// Update config
	newModel := Model{
		Title:    "stacklok-hosted",
		Provider: "vllm",
		Model:    "Qwen/Qwen2.5-Coder-14B-Instruct",
		APIKey:   "key",
		APIBase:  "http://localhost:8989/vllm",
	}

	// Check if model already exists
	modelExists := false
	for i, model := range config.Models {
		if model.Title == "stacklok-hosted" {
			config.Models[i] = newModel
			modelExists = true
			break
		}
	}

	if !modelExists {
		config.Models = append(config.Models, newModel)
	}

	// Update tab autocomplete model
	config.TabAutocompleteModel = &Model{
		Title:    "stacklok-hosted",
		Provider: "vllm",
		Model:    "Qwen/Qwen2.5-Coder-14B-Instruct",
		APIKey:   "",
		APIBase:  "http://localhost:8989/vllm",
	}

	// Write updated config
	configJSON, err := json.MarshalIndent(config, "", "    ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configFile, configJSON, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	fmt.Println("Configuration updated successfully!")
	return nil
}

func setupDocker() error {
	fmt.Println("Checking if Docker is installed...")

	// Check Docker Compose
	composeCmd := "docker"
	composeArgs := []string{"compose"}

	if err := exec.Command("docker", "compose", "version").Run(); err != nil {
		if _, err := exec.LookPath("docker-compose"); err != nil {
			return fmt.Errorf("neither Docker Compose nor docker-compose is installed")
		}
		composeCmd = "docker-compose"
		composeArgs = []string{}
	}

	fmt.Println("Creating docker-compose.yml file...")
	if err := createDockerComposeFile(); err != nil {
		return fmt.Errorf("failed to create docker-compose file: %w", err)
	}

	fmt.Println("Starting Docker containers...")
	cmd := exec.Command(composeCmd, append(composeArgs, "up", "-d")...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start containers: %s: %w", string(output), err)
	}

	fmt.Println("Containers started successfully.")
	fmt.Println("\nYou can now open Visual Studio Code and start using the Codegate extension.")
	fmt.Println("If you have any issues, please check the logs of the containers using 'docker logs <container-name>'.")
	fmt.Println("\nLast of all, you will need a key to use the stacklok inference model, please contact stacklok for a key.")

	return nil
}

func copyFile(src, dst string) error {
	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destination.Close()

	_, err = io.Copy(destination, source)
	return err
}

func createDockerComposeFile() error {
	composeContent := `version: "3.9"

services:
  codegate-proxy:
    networks:
      - codegatenet
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: ghcr.io/stacklok/codegate:latest
    pull_policy: always
    ports:
      - 8989:8989
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command:
      - -vllm=https://inference.codegate.ai
      - -ollama-embed=http://host.docker.internal:11434
      - -package-index=/opt/rag-in-a-box/data/
      - -db=rag-db
    depends_on:
      - rag-qdrant-db

  rag-qdrant-db:
    image: ghcr.io/stacklok/codegate/qdrant-codegate@sha256:fccd830f8eaf9079972fee1eb95908ffe42d4571609be8bffa32fd26610481f7
    container_name: rag-db
    ports:
      - "6333:6333"
      - "6334:6334"
    networks:
      - codegatenet

networks:
  codegatenet:
    driver: bridge`

	return os.WriteFile("docker-compose.yml", []byte(composeContent), 0644)
}
