import httpx
import httpx2
from pydantic_ai.direct import model_request_sync
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from system_one_adapter import SystemOneAdapterClient
from system_one_adapter.providers import ProviderResult
from system_one_adapter.providers.base import translating
from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeError,
)

from .question_types import SystemOneResponse, question_record


class ConfiguredProvider:
    """PydanticAI handles model settings; the adapter handles answer schemas and retries."""

    def __init__(self, model):
        self.model = model
        self.model_name = model.model_name

    def request(self, messages, *, schema, structured):
        with translating(self.translate_error):
            return self._request(messages, schema=schema, structured=structured)

    def _request(self, messages, *, schema, structured):
        if not structured:
            raise ValueError("Workflow evaluations require structured probability answers")
        history = []
        for message in messages:
            if message.role == "assistant":
                history.append(ModelResponse(parts=[TextPart(message.content)]))
            else:
                part = SystemPromptPart if message.role == "system" else UserPromptPart
                history.append(ModelRequest(parts=[part(message.content)]))
        response = model_request_sync(
            self.model,
            history,
            model_request_parameters=ModelRequestParameters(
                output_mode="native",
                output_object=OutputObjectDefinition(json_schema=schema, name="evaluation", strict=True),
            ),
            instrument=False,
        )
        if response.finish_reason in ("length", "content_filter", "error"):
            raise TypeSafeError(f"Model response did not complete: {response.finish_reason}")
        return ProviderResult(
            text="".join(part.content for part in response.parts if isinstance(part, TextPart)),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def translate_error(self, error):
        if isinstance(error, TypeSafeError):
            return error
        if isinstance(error, ModelHTTPError):
            return TypeSafeAPIError(error.status_code, error.body, httpx2.Headers(error.headers or {}))
        cause = error
        while cause is not None:
            if isinstance(cause, (httpx.TimeoutException, httpx2.TimeoutException)):
                return TypeSafeAPITimeoutError(httpx2.Timeout(None))
            if isinstance(cause, (httpx.RequestError, httpx2.RequestError)):
                return TypeSafeAPIConnectionError(str(error))
            cause = cause.__cause__
        return TypeSafeError(str(error))


class QuestionClient:

    def __init__(self, config):
        self.typesafe = config.provider == "typesafe"
        retry = RetryPolicy(max_retries=5, backoff_initial=2.0, backoff_max=60.0, timeout=None)
        if self.typesafe:
            self.client = TypeSafeClient(base_url=config.base_url, timeout=config.timeout_s, retry=retry)
        else:
            self.client = SystemOneAdapterClient(
                structured_outputs=True,
                llm_answer_mode="probabilities",
                normalize_probabilities=True,
                n_retry_malformed_structure=config.malformed_retries,
                retry=retry,
            )

    def system_one(self, *, model, state, questions) -> SystemOneResponse:
        return self.client.system_one(
            state=state,
            questions={key: question_record(question) for key, question in questions.items()},
            model=model if self.typesafe else ConfiguredProvider(model),
        )

    def close(self):
        self.client.close()
