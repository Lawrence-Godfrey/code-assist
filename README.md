# Code Assistant

A tool for code analysis and generation using large language models.

## Installation

```bash
pip install code-assistant
```

## Usage

The package provides a command-line interface with several subcommands:

### Extract Code from GitHub

```bash
# Extract code from a GitHub repository
code-assistant extract github \
    --repo-url="https://github.com/username/repo" \
    --output-path="code_units.json" \
    --github-token="your_token"  # Optional, for private repos
```

### Generate Embeddings

```bash
# Generate embeddings for code units
code-assistant embed generate \
    --input-path="code_units.json" \
    --output-path="embedded_code_units.json" \
    --model-name="jinaai/jina-embeddings-v3"

# Compare a query against embedded code
code-assistant embed compare \
    --query="How do I handle errors?" \
    --input-path="embedded_code_units.json" \
    --model-name="jinaai/jina-embeddings-v3"
```

### Generate Training Data

```bash
# Generate prompt-code pairs using OpenAI
code-assistant generate prompts \
    --code-units-path="code_units.json" \
    --output-path="prompt_code_pairs.json" \
    --num-rows=100
```

### Evaluate Retrieval

```bash
# Evaluate retrieval performance
code-assistant evaluate retrieval \
    --test-data-path="test_prompt_code_pairs.json" \
    --codebase-path="embedded_code_units.json" \
    --output-path="evaluation_results.json"
```

## Environment Variables

The following environment variables can be set:

- `GITHUB_TOKEN`: GitHub personal access token for private repository access
- `OPENAI_API_KEY`: OpenAI API key for generating prompts and embeddings

## Development

To set up the development environment:

```bash
# Install in development mode
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.