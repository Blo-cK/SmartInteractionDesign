# Gwen Human Description Service

Consumes person images from `human_image_capture` service and generates natural language descriptions using Ollama vision models.

## Prerequisites

### Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download installer from [ollama.com/download](https://ollama.com/download)

### Download Vision Model

```bash
ollama pull qwen3-vl:4b
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit configuration at the top of the script:

- `CONSUME_SOURCES` (["camera1"]): List of source IDs to consume images from
- `CONSUME_SERVICE` ("human_image_capture"): Service ID to consume from
- `SYSTEM_PROMPT`: Instructions for Ollama what to describe (person details, clothing, features)
- `kafka_broker` ("152.53.32.66:9094"): Kafka broker address
- `ollama_url` ("http://localhost:11434"): Ollama API endpoint
- `ollama_model` ("qwen3-vl:4b"): Vision model to use
- `max_retries` (3): Max retry attempts for Ollama API calls
- `system_prompt`: Custom prompt for description generation

## Output

- **Output**: `output.<source_id>.human_description` (Natural language descriptions)
- **Storage**: Descriptions saved to `descriptions/descriptions_<source_id>.jsonl`


