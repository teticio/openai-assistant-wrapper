import logging
import time
from typing import AsyncGenerator

import openai
from fastapi import FastAPI, Header, HTTPException, Response
from openai.types.beta.assistant_stream_event import (
    ThreadMessageCompleted,
    ThreadMessageDelta,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.completion_create_params import (
    CompletionCreateParams,
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from starlette.responses import StreamingResponse

app = FastAPI()

logger = logging.getLogger(__name__)


@app.post(
    "/api/providers/openai/v1/chat/completions",
    response_model=bytes | ChatCompletion,
)
async def chat_completions(
    request: CompletionCreateParams, authorization: str = Header(...)
) -> StreamingResponse | Response:
    """
    Endpoint for creating a chat completion stream using OpenAI's API.

    Args:
        request (CompletionCreateParamsStreaming): Parameters for creating a streaming chat completion.
        authorization (str): Bearer token for authorizing the API request.

    Returns:
        StreamingResponse | Response: A StreamingResponse containing the chat completion stream or a Response object.

    Raises:
        HTTPException: If an error occurs during processing the request.
    """
    try:
        api_key = None
        if authorization.startswith("Bearer "):
            api_key = authorization[len("Bearer ") :].strip()
        client = openai.OpenAI(api_key=api_key)

        if "stream" in request and request["stream"]:
            return await chat_completions_streaming(client, request)

        return await chat_completions_non_streaming(client, request)

    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def chat_completions_streaming(
    client: openai.OpenAI,
    request: CompletionCreateParamsStreaming,
) -> StreamingResponse:
    """
    Endpoint for creating a chat completion stream using OpenAI's API.

    Args:
        request (CompletionCreateParamsStreaming): Parameters for creating a streaming chat completion.

    Returns:
        StreamingResponse: An async generator yielding streamed chunks of chat completion data.

    Raises:
        HTTPException: If an error occurs during processing the request.
    """
    try:
        thread = client.beta.threads.create()
        for message in request["messages"]:
            if message["role"] != "system":
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role=message["role"],
                    content=message["content"],
                )

        created = int(time.time())
        stream = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=request["model"],
            stream=True,
        )

        async def event_generator() -> AsyncGenerator[bytes, None]:
            """
            Asynchronous generator that yields event data from the OpenAI stream.

            Yields:
                bytes: JSON-formatted chat completion chunks encoded as bytes.
            """
            for event in stream:
                if isinstance(event, ThreadMessageDelta):
                    choices = [
                        ChunkChoice(
                            delta=ChoiceDelta(content=choice.text.value),
                            index=index,
                        )
                        for index, choice in enumerate(event.data.delta.content)
                    ]

                    yield (
                        "data: "
                        + ChatCompletionChunk(
                            id=event.data.id,
                            choices=choices,
                            created=created,
                            model=request["model"],
                            object="chat.completion.chunk",
                        ).model_dump_json()
                        + "\n\n"
                    ).encode("utf-8")

                elif isinstance(event, ThreadMessageCompleted):
                    choices = [
                        ChunkChoice(
                            delta=ChoiceDelta(content=""), index=0, finish_reason="stop"
                        )
                    ]
                    yield (
                        "data: "
                        + ChatCompletionChunk(
                            id=event.data.id,
                            choices=choices,
                            created=created,
                            model=request["model"],
                            object="chat.completion.chunk",
                        ).model_dump_json()
                        + "\n\ndata: [DONE]\n\n"
                    ).encode("utf-8")

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def chat_completions_non_streaming(
    client: openai.OpenAI,
    request: CompletionCreateParamsNonStreaming,
) -> Response:
    """
    Endpoint for creating a chat completion response using OpenAI's API.

    Args:
        request (CompletionCreateParamsNonStreaming): Parameters for creating a non-streaming chat completion.

    Returns:
        Response: a response containing the chat completion data.

    Raises:
        HTTPException: If an error occurs during processing the request.
    """
    try:
        thread = client.beta.threads.create()
        for message in request["messages"]:
            if message["role"] != "system":
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role=message["role"],
                    content=message["content"],
                )

        created = int(time.time())
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=request["model"],
        )
        assert run.status == "completed"
        message = next(iter(client.beta.threads.messages.list(thread_id=thread.id)))

        choices = [
            Choice(
                index=index,
                finish_reason="stop",
                message=ChatCompletionMessage(content=choice.text.value, role="assistant"),
            )
            for index, choice in enumerate(message.content)
        ]

        return Response(
            ChatCompletion(
                id=message.id,
                choices=choices,
                created=created,
                model=request["model"],
                object="chat.completion",
            ).model_dump_json(),
            media_type="application/json",
        )

    except HTTPException as e:
        logger.error(f"HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
