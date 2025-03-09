import base64
import io
import logging
import os
import time
import uuid
from typing import AsyncGenerator, Iterable

from fastapi import FastAPI, Header, HTTPException, Response
from openai import AsyncOpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.threads.message import Message
from openai.types.beta.assistant_stream_event import (
    ThreadMessageCompleted,
    ThreadMessageDelta,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import (
    CompletionCreateParams,
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from starlette.responses import StreamingResponse

app = FastAPI()

logger = logging.getLogger(__name__)


async def create_messages(
    client: AsyncOpenAI, thread: Thread, messages: Iterable[ChatCompletionMessageParam]
) -> None:
    """
    Process a list of chat messages and create them in a specified thread.

    This function iterates over a collection of chat messages, processes each message,
    and creates it in the specified thread using the OpenAI client. If a message contains
    an image URL in base64 format, it decodes the image, uploads it to the OpenAI service,
    and replaces the URL with a file reference.

    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client used to interact with the API.
        thread (Thread): The thread in which the messages will be created.
        messages (Iterable[ChatCompletionMessageParam]): An iterable of message parameters
            to be processed and created in the thread.

    Returns:
        None
    """
    for message in messages:
        if message["role"] == "system":
            continue

        content = message["content"]
        if not isinstance(message["content"], str):
            content = list(content)
            for i, content_item in enumerate(content):
                if content_item["type"] == "image_url" and content_item["image_url"][
                    "url"
                ].startswith("data"):
                    header, encoded = content_item["image_url"]["url"].split(",", 1)
                    file_extension = header.split(";")[0].split("/")[-1]
                    image_data = base64.b64decode(encoded)
                    file_obj = io.BytesIO(image_data)
                    file_obj.name = f"{uuid.uuid4()}.{file_extension}"
                    file = await client.files.create(file=file_obj, purpose="vision")
                    content[i] = {
                        "type": "image_file",
                        "image_file": {"file_id": file.id},
                    }

        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role=message["role"],
            content=content,
        )


async def get_references(client: AsyncOpenAI, message: Message) -> str:
    """
    Add references based on citations from a thread.

    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client.
        message (Message): The message object containing content and annotations.

    Returns:
        str: A string containing the references for the thread.
    """
    if os.getenv("DO_NOT_USE_REFERENCES"):
        return ""

    citations = {}
    annotations = [
        annotation
        for content in message.content
        for annotation in content.text.annotations
    ]
    for annotation in annotations:
        if file_citation := getattr(annotation, "file_citation", None):
            citations[annotation.text] = await client.files.retrieve(
                file_citation.file_id
            )

    references = "\n".join(
        f"- {citation}: {file.filename}" for citation, file in citations.items()
    )
    if references:
        references = f"\n\n**References**:\n{references}"
    return references


@app.post(
    "/api/providers/openai/v1/chat/completions",
    response_model=bytes | ChatCompletion,
)
async def chat_completions(
    request: CompletionCreateParams, authorization: str = Header(...)
) -> StreamingResponse | Response:
    """
    Create a chat completion stream using OpenAI's API.

    Args:
        request (CompletionCreateParams): Parameters for creating a chat completion.
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
        client = AsyncOpenAI(api_key=api_key)

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
    client: AsyncOpenAI,
    request: CompletionCreateParamsStreaming,
) -> StreamingResponse:
    """
    Create a chat completion stream using OpenAI's API.

    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client.
        request (CompletionCreateParamsStreaming): Parameters for creating a streaming chat completion.

    Returns:
        StreamingResponse: An async generator yielding streamed chunks of chat completion data.

    Raises:
        HTTPException: If an error occurs during processing the request.
    """
    try:
        thread = await client.beta.threads.create()
        await create_messages(client, thread, request["messages"])
        created = int(time.time())
        stream = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=request["model"],
            stream=True,
        )

        async def event_generator() -> AsyncGenerator[bytes, None]:
            """
            Asynchronously generate event data from the OpenAI stream.

            Yields:
                bytes: JSON-formatted chat completion chunks encoded as bytes.
            """
            async for event in stream:
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
                    messages = await client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    message = await anext(aiter(messages))
                    references = await get_references(client, message)
                    disclaimer = os.getenv("DISCLAIMER", "")
                    if any(
                        disclaimer in content_item.text.value for content_item in message.content
                    ):
                        disclaimer = ""
                    choices = [
                        ChunkChoice(
                            delta=ChoiceDelta(content=references + disclaimer),
                            index=0,
                            finish_reason="stop",
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
    client: AsyncOpenAI,
    request: CompletionCreateParamsNonStreaming,
) -> Response:
    """
    Create a chat completion response using OpenAI's API.

    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client.
        request (CompletionCreateParamsNonStreaming): Parameters for creating a non-streaming chat completion.

    Returns:
        Response: A response containing the chat completion data.

    Raises:
        HTTPException: If an error occurs during processing the request.
    """
    try:
        thread = await client.beta.threads.create()
        await create_messages(client, thread, request["messages"])
        created = int(time.time())
        run = await client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=request["model"],
        )
        assert run.status == "completed"

        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        message = await anext(aiter(messages))
        references = await get_references(client, message)
        disclaimer = os.getenv("DISCLAIMER", "")
        if any(
            disclaimer in content_item.text.value for content_item in message.content
        ):
            disclaimer = ""
        choices = [
            Choice(
                index=index,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content=choice.text.value + references + disclaimer,
                    role="assistant",
                ),
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
