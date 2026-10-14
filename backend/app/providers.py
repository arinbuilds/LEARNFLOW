"""LLM provider adapter. Prompts and business rules stay provider-independent."""

import asyncio
import json

from .config import Settings


class LLMProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, system: str, prompt: str, max_tokens: int = 650) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "mock":
            return self._offline_answer(system, prompt)
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            response = await client.chat.completions.create(
                model=self.settings.llm_model or "gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        if provider == "gemini":
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.settings.gemini_api_key)
            response = await client.aio.models.generate_content(
                model=self.settings.llm_model or "gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=max_tokens, system_instruction=system),
            )
            return response.text or ""
        if provider == "bedrock":
            import boto3

            def invoke():
                client = boto3.client("bedrock-runtime", region_name=self.settings.aws_region)
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                })
                response = client.invoke_model(
                    modelId=self.settings.llm_model or self.settings.bedrock_model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                data = json.loads(response["body"].read())
                return "".join(part.get("text", "") for part in data.get("content", []))

            return await asyncio.to_thread(invoke)
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    @staticmethod
    def _offline_answer(system: str, prompt: str) -> str:
        """A deterministic local demo responder; no external model or unsupported claims."""
        if "NEXT_INTERVIEW_QUESTION" in prompt:
            if "PREVIOUS_ANSWER:" in prompt:
                answer = prompt.split("PREVIOUS_ANSWER:", 1)[1].split("\n", 1)[0].strip()
                if answer:
                    seed = answer[:90]
                    return f"You mentioned {seed!r}. What trade-off did you consider, and how would you validate that choice?"
            role = prompt.split("ROLE:", 1)[1].split("\n", 1)[0].strip() if "ROLE:" in prompt else "this role"
            return f"Tell me about a project or experience that best prepares you for the {role} role. What was your contribution and what outcome did you achieve?"
        if "SCORECARD_JSON" in prompt:
            return "{}"
        if "GROUNDED_LEARNING_ANSWER" in prompt:
            context = prompt.split("ALLOWED SOURCES:", 1)[-1].strip()
            if not context:
                return "I couldn't find support for that in the course resources available at this point."
            source_lines = [line for line in context.splitlines() if line.startswith("[S")]
            first = source_lines[0] if source_lines else context.split("\n", 1)[0]
            body = first.split(": ", 1)[-1]
            action = prompt.split("ACTION:", 1)[-1].split("\n", 1)[0].strip().lower() if "ACTION:" in prompt else ""
            if action.startswith("create three short"):
                return "Practice with this passage:\n" + body[:600] + "\n\n1. What is its main idea?\n2. Explain it in your own words.\n3. Which statement in the passage supports your explanation?\n\nUse the cited passage to check your answers. " + first.split(": ", 1)[0]
            if action.startswith("summarize"):
                bullets = "\n".join(f"- {line.split(': ', 1)[-1][:230]}" for line in source_lines[:5])
                return f"Here's a short recap of the material available so far:\n{bullets}"
            if action.startswith("compare") and len(source_lines) > 1:
                return f"Compare these two passages from the lesson. First: {source_lines[0].split(': ', 1)[-1][:230]}\nSecond: {source_lines[1].split(': ', 1)[-1][:230]}"
            if action.startswith("explain in simpler"):
                return f"In simpler terms: {body[:430]} " + first.split(": ", 1)[0]
            if action.startswith("give one concrete"):
                return f"Here is the relevant passage: {body[:600]} {first.split(': ', 1)[0]}\n\nDemo mode quotes the transcript. Configure an AI provider for a tailored example."
            if action.startswith("point to where"):
                return f"This idea is covered in {first.split(': ', 1)[0]}. The supporting passage is: {body[:350]}"
            return f"Based on the available course material: {body[:500]} {first.split(': ', 1)[0]}"
        return "Use the course material shown below as evidence, and ask one focused follow-up question."
