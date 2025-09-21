from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import urllib.request
import os
import json
import vim

if "VIMAI_DUMMY_IMPORT" in os.environ:
    from py.types import AIMessage, AIResponseChunk, AIImageResponseChunk, AIUtils, AIProvider, AICommandType


class OpenAICodexProvider:
    default_options_varname_complete = "g:vim_ai_openai_codex_complete"
    default_options_varname_edit = "g:vim_ai_openai_codex_edit"

    def __init__(self, command_type: "AICommandType", raw_options: Mapping[str, str], utils: "AIUtils") -> None:
        self.utils = utils
        self.command_type = command_type
        config_varname = getattr(self, f"default_options_varname_{command_type}", "")
        raw_default_options = vim.eval(config_varname) if config_varname else {}
        merged_options = {**raw_default_options, **raw_options}
        self.options = self._parse_raw_options(merged_options)
        self._ensure_supported_command()

    def _ensure_supported_command(self) -> None:
        if self.command_type not in ("complete", "edit"):
            raise self.utils.make_known_error("OpenAI Codex provider supports only :AI and :AIEdit commands")

    def request(self, messages: "Sequence[AIMessage]") -> Iterator["AIResponseChunk"]:
        prompt = self._messages_to_prompt(messages)
        options = self.options
        http_options = {
            "request_timeout": options.get("request_timeout") or 20,
            "auth_type": options["auth_type"],
            "token_file_path": options["token_file_path"],
            "token_load_fn": options["token_load_fn"],
        }
        openai_options = self._make_codex_options(options)
        request = {"prompt": prompt, **openai_options}

        self.utils.print_debug("openai-codex: [{}] request: {}", self.command_type, request)
        url = options["endpoint_url"]
        response = self._openai_request(url, request, http_options)

        def _map_chunk(resp: Mapping[str, Any]) -> "AIResponseChunk | None":
            self.utils.print_debug("openai-codex: [{}] response: {}", self.command_type, resp)
            choices = resp.get("choices") or [{}]
            text = choices[0].get("text", "")
            if text:
                return {"type": "assistant", "content": text}
            return None

        return filter(None, map(_map_chunk, response))

    def request_image(self, prompt: str) -> list["AIImageResponseChunk"]:
        raise self.utils.make_known_error("OpenAI Codex provider does not support image generation")

    # -- prompt helpers -------------------------------------------------

    @staticmethod
    def _messages_to_prompt(messages: "Sequence[AIMessage]") -> str:
        role_labels = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }
        prompt_parts: list[str] = []
        for message in messages:
            text_chunks = [c.get("text", "") for c in message.get("content", []) if c.get("type") == "text"]
            text = "\n".join(filter(None, text_chunks)).strip()
            if not text:
                continue
            role = role_labels.get(message.get("role", "user"), "User")
            prompt_parts.append(f"{role}::\n{text}")
        if not prompt_parts or not prompt_parts[-1].startswith("Assistant::"):
            prompt_parts.append("Assistant::")
        return "\n\n".join(prompt_parts)

    # -- option helpers -------------------------------------------------

    def _parse_raw_options(self, raw_options: Mapping[str, Any]) -> dict[str, Any]:
        options = dict(raw_options)

        def _convert_option(name: str, converter) -> None:
            if name in options and isinstance(options[name], str) and options[name] != "":
                try:
                    options[name] = converter(options[name])
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    raise self.utils.make_known_error(
                        f"Invalid value for option '{name}': {options[name]}. Error: {error}"
                    )

        _convert_option("request_timeout", float)
        _convert_option("temperature", float)
        _convert_option("max_tokens", int)
        _convert_option("top_p", float)
        _convert_option("presence_penalty", float)
        _convert_option("frequency_penalty", float)
        _convert_option("best_of", int)
        _convert_option("n", int)
        _convert_option("stream", lambda x: bool(int(x)))
        _convert_option("logprobs", int)
        _convert_option("suffix", str)
        _convert_option("stop", json.loads)
        _convert_option("logit_bias", json.loads)

        return options

    def _make_codex_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": options["model"],
        }

        option_keys = [
            "suffix",
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stream",
            "logprobs",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "best_of",
            "logit_bias",
        ]

        for key in option_keys:
            if key not in options:
                continue
            value = options[key]
            if value in ("", None):
                continue
            if key in {"max_tokens", "logprobs", "best_of", "n"} and value == 0:
                continue
            result[key] = value

        return result

    # -- HTTP helpers ---------------------------------------------------

    def _load_api_key(self) -> tuple[str, str | None]:
        raw_api_key = self.utils.load_api_key(
            "OPENAI_API_KEY",
            token_file_path=self.options.get("token_file_path", ""),
            token_load_fn=self.options.get("token_load_fn", ""),
        )
        elements = raw_api_key.strip().split(",")
        api_key = elements[0].strip()
        org_id = elements[1].strip() if len(elements) > 1 else None
        return api_key, org_id

    def _openai_request(self, url: str, data: Mapping[str, Any], options: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        resp_data_prefix = "data: "
        resp_done = "[DONE]"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VimAI",
        }

        auth_type = options["auth_type"]
        if auth_type == "bearer":
            api_key, org_id = self._load_api_key()
            headers["Authorization"] = f"Bearer {api_key}"
            if org_id:
                headers["OpenAI-Organization"] = org_id
        elif auth_type == "api-key":
            api_key, _ = self._load_api_key()
            headers["api-key"] = api_key

        request_timeout = options.get("request_timeout", 20)
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=request_timeout) as response:
            if not data.get("stream"):
                yield json.loads(response.read().decode())
                return
            for line_bytes in response:
                line = line_bytes.decode("utf-8", errors="replace")
                if not line.startswith(resp_data_prefix):
                    continue
                payload = line[len(resp_data_prefix):].strip()
                if payload == resp_done:
                    continue
                yield json.loads(payload)
