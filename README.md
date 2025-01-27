<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/codegate-logo-white.svg">
  <img alt="CodeGate logo" src="./static/codegate-logo-dark.svg" width="800px" style="max-width: 100%;">
</picture>

---

[![CI](https://github.com/stacklok/codegate/actions/workflows/run-on-push.yml/badge.svg)](https://github.com/stacklok/codegate/actions/workflows/run-on-push.yml)
|
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
|
[![Discord](https://dcbadge.vercel.app/api/server/RkzVuTp3WK?logo=discord&label=Discord&color=5865&style=flat)](https://discord.gg/RkzVuTp3WK)

---

# CodeGate: Secure AI Coding Assistance

**By [Stacklok](https://stacklok.com)**  

CodeGate is a **local gateway** that makes AI coding assistants safer. It ensures AI-generated recommendations adhere to best practices while safeguarding your code's integrity and protecting your privacy. With CodeGate, you can confidently leverage AI in your development workflow without sacrificing security or productivity.

---

## ✨ Why Choose CodeGate?
AI coding assistants are powerful, but they can inadvertently introduce risks. CodeGate protects your development process by:

- 🔒 **Preventing accidental exposure of secrets and sensitive data**
- 🛡️ **Ensuring AI suggestions follow secure coding practices**
- ⚠️ **Blocking recommendations of known malicious or deprecated libraries**
- 🔍 **Providing real-time security analysis of AI suggestions**

---

## 🚀 Quickstart

### Prerequisites
CodeGate is distributed as a Docker container. You need a container runtime like Docker Desktop or Docker Engine. **Podman** and **Podman Desktop** are also supported. CodeGate works on **Windows**, **macOS**, and **Linux** operating systems with **x86_64** and **arm64** (ARM and Apple Silicon) CPU architectures.

These instructions assume the `docker` CLI is available. If you use Podman, replace `docker` with `podman` in all commands.

### Installation
To start CodeGate, run this simple command:

```bash
docker run --name codegate -d -p 8989:8989 -p 9090:9090 -p 8990:8990 \
  --mount type=volume,src=codegate_volume,dst=/app/codegate_volume \
  --restart unless-stopped ghcr.io/stacklok/codegate:latest
```

That’s it! CodeGate is now running locally. For advanced configurations and parameter references, check out the [CodeGate Install and Upgrade](https://docs.codegate.ai/how-to/install) documentation.

---

## 🖥️ Dashboard
CodeGate includes a web dashboard that provides:
- A view of **security risks** detected by CodeGate
- A **history of interactions** between your AI coding assistant and your LLM

### Accessing the Dashboard
Ensure port `9090` is bound to a port on your local system when launching CodeGate. For example:

```bash
docker run --name codegate -d -p 8989:8989 \
  -p 9090:9090 \
  --restart unless-stopped ghcr.io/stacklok/codegate:latest
```

Once CodeGate is running, open [http://localhost:9090](http://localhost:9090) in your web browser to access the dashboard.

To learn more, visit the [CodeGate Dashboard documentation](https://docs.codegate.ai/how-to/dashboard).

---

## 🔐 Features

### Secret Encryption
CodeGate helps you protect sensitive information from being accidentally exposed to AI models and third-party AI provider systems by redacting detected secrets from your prompts using encryption. [Learn more](https://docs.codegate.ai/features/secrets-encryption)

### Dependency Risk Awareness
LLMs’ knowledge cutoff date is often months or even years in the past. They might suggest outdated, vulnerable, or non-existent packages (hallucinations), exposing you and your users to security risks. 

CodeGate scans direct, transitive, and development dependencies in your package definition files, installation scripts, and source code imports that you supply as context to an LLM. [Learn more](https://docs.codegate.ai/features/dependency-risk)

### Security Reviews
CodeGate performs security-centric code reviews, identifying insecure patterns or potential vulnerabilities to help you adopt more secure coding practices. [Learn more](https://docs.codegate.ai/features/security-reviews)

---

## 🤖 Supported AI Coding Assistants and Providers

### [Aider](https://aider.chat)
- **Local / Self-Managed:**
  - [Ollama](https://ollama.com/)
- **Hosted:**
  - [OpenAI API](https://openai.com/api/)

🔥 Getting Started with CodeGate and aider [Watch on YouTube](https://www.youtube.com/watch?v=VxvEXiwEGnA)

### [Continue](https://www.continue.dev/)
- **Local / Self-Managed:**
  - [Ollama](https://ollama.com/)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp)
  - [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- **Hosted:**
  - [OpenRouter](https://openrouter.ai/)
  - [Anthropic API](https://www.anthropic.com/api)
  - [OpenAI API](https://openai.com/api/)

### [GitHub Copilot](https://github.com/features/copilot)
- The Copilot plugin works with **Visual Studio Code (VS Code)**. 
- **Support for JetBrains is coming soon.**

---

## 🛡️ Privacy First
Unlike other tools, with CodeGate **your code never leaves your machine**. CodeGate is built with privacy at its core:

- 🏠 **Everything stays local**
- 🚫 **No external data collection**
- 🔐 **No calling home or telemetry**
- 💪 **Complete control over your data**

---

## 🛠️ Development
Are you a developer looking to contribute? Dive into our technical resources:

- [Development Guide](https://github.com/stacklok/codegate/blob/main/docs/development.md)
- [CLI Commands and Flags](https://github.com/stacklok/codegate/blob/main/docs/cli.md)
- [Configuration System](https://github.com/stacklok/codegate/blob/main/docs/configuration.md)
- [Logging System](https://github.com/stacklok/codegate/blob/main/docs/logging.md)

---

## 🤝 Contributing
We welcome contributions! Whether you're submitting bug reports, feature requests, or code contributions, your input makes CodeGate better for everyone. We thank you ❤️!

Start by reading our [Contributor Guidelines](https://github.com/stacklok/codegate/blob/main/CONTRIBUTING.md).

---

## 🌟 Support Us
Love CodeGate? Starring this repository and sharing it with others helps CodeGate grow 🌱

[![Star on GitHub](https://img.shields.io/github/stars/stacklok/codegate.svg?style=social)](https://github.com/stacklok/codegate)

## 📜 License
CodeGate is licensed under the terms specified in the [LICENSE file](https://github.com/stacklok/codegate/blob/main/LICENSE).

---

<!-- markdownlint-disable-file first-line-heading no-inline-html -->
