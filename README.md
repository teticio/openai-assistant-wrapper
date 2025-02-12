# OpenAI Assistant Wrapper

## Description
Wraps the OpenAI Assistant API with the standard OpenAI `/chat/completions` API for both streaming and non-streaming responses. This can be useful for deploying your Assistants in any application which conforms to the OpenAI API standard.

## Installation
```bash
git clone https://github.com/teticio/openai-assistant-wrapper.git
cd openai-assistant-wrapper
poetry install --no-root
```

## Usage
Create a `.env` file with

```bash
OPENAI_API_KEY=your-api-key  # or pass the API key in the request
OPENAI_BASE_URL=http://localhost:8000/api/providers/openai/v1
```

and run

```bash
poetry run .venv/bin/uvicorn main:app --reload
```

In the API call, provide your Assistant ID (e.g., `asst_1234567890abcdef01234567`) as the `model`. The [`test.ipynb`](notebooks/test.ipynb) notebook gives some examples of how to use the wrapper.
