"""LiteLLM streaming client for generating answers from retrieved evidence."""

import logging
from typing import AsyncIterator, Optional

from litellm import acompletion

from mcp_server.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Streaming client for LiteLLM ChatCompletion API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._api_url = (api_url or settings.LITELLM_API_URL).rstrip("/")
        self._api_key = api_key or settings.LITELLM_API_KEY
        self._model = model or settings.LITELLM_MODEL

    async def stream_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream LLM answer from retrieved context.

        Args:
            query: User's question.
            context: Formatted evidence context from retriever.
            system_prompt: Optional system message.

        Yields:
            Text chunks of the LLM response.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a knowledge base assistant. Answer the user's question "
                "based solely on the provided context. If the context doesn't "
                "contain relevant information, say so clearly.\n\n"
                "CITATION RULES:\n"
                "- Use bracketed numbers matching the Sources list, e.g. [1], [2].\n"
                "- Each [Source: n] tag in the context corresponds to the numbered "
                "entry in the Sources section.\n"
                "- At the end of your answer, include a Sources section listing "
                "only the references you actually cited.\n"
                "- Be precise: cite the specific source number, not just a title."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Please answer based on the context above."
                ),
            },
        ]

        try:
            response = await acompletion(
                model=self._model,
                api_base=self._api_url,
                api_key=self._api_key,
                messages=messages,
                stream=True,
                temperature=0.3,
                max_tokens=4096,
                timeout=60,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"\n[Error generating answer: {e}]"

    async def answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Get a non-streaming answer.

        Args:
            query: User's question.
            context: Formatted evidence context.
            system_prompt: Optional system message.

        Returns:
            Complete answer text.
        """
        parts = []
        async for chunk in self.stream_answer(query, context, system_prompt):
            parts.append(chunk)
        return "".join(parts)