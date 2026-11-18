import argparse
import datetime
import httpx
import json
import openai
import os
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
import secrets
import time

# Same eval as endgame_anthropic.py (prompt and helpers are imported from it),
# but talks to an OpenAI-compatible Chat Completions endpoint.
# Built for providers that only expose /chat/completions and take a numeric
# reasoning effort there, e.g. Tinker:
#   --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
#   --reasoning-effort 0.5
from endgame_anthropic import (
    API_TIMEOUT, READ_TIMEOUT, N_ATTEMPTS_MAX, RETRYABLE_STATUS,
    COLORS, TOOL_DESCRIPTION, ApiFailure, format_prompt, shuffle_merge,
)

TOOLS = [{
    "type" : "function",
    "function" : {
        "name" : "make_guesses",
        "description" : TOOL_DESCRIPTION,
        "parameters" : {
            "type" : "object",
            "properties" : {
                "guesses": {
                    "type" : "array",
                    "items" : {
                        "type": "object",
                        "properties": {
                            "codemaker": {"type": "string"},
                            "guess": {"type": "string"}
                        },
                        "required": ["codemaker", "guess"],
                        "additionalProperties": False
                    },
                },
            },
            "required" : ["guesses"],
            "additionalProperties": False
        },
    },
}]


def parse_effort(s):
    """Numbers go over the wire as numbers (0.5), anything else as a named level ("high")."""
    try:
        return float(s)
    except ValueError:
        return s


def save_error_response(response):
    os.makedirs("logs/errors", exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    path = f"logs/errors/{dt}-{secrets.token_hex(3)}.json"
    with open(path, "x") as fw:
        fw.write(json.dumps(response))
    print(f"W: saved full response to {path}")
    return path


def collect_stream(chunks):
    """Fold a chat.completions stream into one plain-dict response."""
    out = {
        "model"         : None,
        "finish_reason" : None,
        "content"       : "",
        "reasoning"     : "",
        "tool_calls"    : [],
        "usage"         : None,
    }
    calls = {}
    for chunk in chunks:
        if chunk.model:
            out["model"] = chunk.model
        if chunk.usage is not None:
            # with include_usage this arrives on the last chunk, which has no choices
            out["usage"] = chunk.usage.model_dump()
        for choice in chunk.choices:
            if choice.finish_reason is not None:
                out["finish_reason"] = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                out["content"] += delta.content
            # not part of the OpenAI schema, so the SDK keeps it as an extra field
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                out["reasoning"] += reasoning
            for tc in delta.tool_calls or []:
                call = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    call["id"] = tc.id
                if tc.function is not None:
                    call["name"] += tc.function.name or ""
                    call["arguments"] += tc.function.arguments or ""
    out["tool_calls"] = [calls[i] for i in sorted(calls)]
    return out


def create_response(client, **kwargs):
    last_err = None
    for attempt in range(1, N_ATTEMPTS_MAX + 1):
        try:
            # Stream to avoid HTTP timeouts on long thinking.
            with client.chat.completions.create(stream=True, **kwargs) as chunks:
                response = collect_stream(chunks)
            if response["finish_reason"] is not None:
                return response, attempt
            last_err = RuntimeError("stream ended without a finish_reason")
        except (openai.APIError, httpx.HTTPError) as e:
            last_err = e
            if isinstance(e, openai.APIStatusError) and e.status_code < 500 and e.status_code not in RETRYABLE_STATUS:
                print(f"W: client error, not retrying: {e}")
                raise ApiFailure(e, attempt)
        print(f"W: attempt {attempt}/{N_ATTEMPTS_MAX} failed: {last_err}")
        if attempt < N_ATTEMPTS_MAX:
            time.sleep(5 * attempt)
    raise ApiFailure(last_err, N_ATTEMPTS_MAX)


def run(client, samples, args):
    n_games = len(samples)
    codemakers = [c for c, _ in COLORS[:n_games]]
    games = {c : s for c, s in zip(codemakers, samples)}
    answers = {c : s['secret'] for c, s in zip(codemakers, samples)}

    records = []
    for c, s in games.items():
        records.append([f"{c} {t['guess']} {t['reply']}" for t in s['turns'][:-1]])
    history = shuffle_merge(records)

    # TODO: we might have different rate for different samples
    prompt = format_prompt(0.2, ",".join(codemakers), "\n".join(history))
    messages = [{"role": "user", "content": prompt}]

    # tool_choice / parallel_tool_calls are left at provider defaults unless asked for:
    # Tinker fails the request with 422 when the reply breaks them.
    extra = {"tool_choice": "required"} if args.tool_choice == "required" else {}

    res = {
        "n_solved"        : 0,
        "n_continuations" : 0,
        "n_tokens_in"     : 0,
        "n_tokens_out"    : 0,
    }

    try:
        response, attempts = create_response(
                client,
                model=args.model,
                messages=messages,
                tools=TOOLS,
                max_completion_tokens=args.n_tokens_max,
                stream_options={"include_usage": True},
                # The SDK types reasoning_effort as a string; extra_body lets a number through as-is.
                extra_body={"reasoning_effort": args.reasoning_effort},
                **extra,
            )
    except ApiFailure as e:
        print(f"W: giving up: {e}")
        res["api_error"] = str(e)
        return res

    res["attempts"] = attempts
    res["served_by"] = response["model"]
    usage = response["usage"]
    if usage is not None:
        res["n_tokens_in"] += usage.get("prompt_tokens") or 0
        res["n_tokens_out"] += usage.get("completion_tokens") or 0
        details = usage.get("completion_tokens_details") or {}
        if details.get("reasoning_tokens") is not None:
            res["n_tokens_reasoning"] = details["reasoning_tokens"]
    else:
        print("W: no usage in response, token counts will be 0")

    res["stop_reason"] = response["finish_reason"]
    if response["finish_reason"] not in ("tool_calls", "stop"):
        # length, content_filter, ...
        print(f"W: response error, finish_reason={response['finish_reason']}")
        res["error_log"] = save_error_response(response)
        res["incomplete_reason"] = response["finish_reason"]
        return res

    # check that we have exactly one guess tool call per instructions
    calls = [c for c in response["tool_calls"] if c["name"] == "make_guesses"]

    if len(calls) != 1:
        print(f"W: Expected one guess tool call per turn, got {len(calls)}")
        res["error_log"] = save_error_response(response)
        return res

    try:
        guesses = json.loads(calls[0]["arguments"])["guesses"]
        res['guesses'] = guesses
        res['answers'] = answers

        for c, a in answers.items():
            gg = [g["guess"] for g in guesses if g["codemaker"] == c]
            if len(gg) != 1:
                pprint(f"W: {c}: {gg}")
                continue
            if gg[0] == a:
                res['n_solved'] += 1

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"E: {e}")
        res["error_log"] = save_error_response(response)
        return res

    return res


def main():
    os.makedirs("logs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", help="Path to specific sample(s), comma separated.")
    parser.add_argument("--samples-file", help="File with one batch per line; each line is comma-separated sample paths.")
    parser.add_argument("--alias", "-a", help="model alias to save in logs.")
    parser.add_argument("--model", "-m", required=True, help="Model name.")
    parser.add_argument("--reasoning-effort", default='low', type=parse_effort,
                        help="Passed as reasoning_effort. A named level (low, medium, high, ...) "
                             "or a number (e.g. 0.5) for providers that accept one.")
    parser.add_argument("--base-url")
    parser.add_argument("--n_tokens_max", type=int, default=2**16, help="max_completion_tokens per request.")
    parser.add_argument("--api-key")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of batches to run concurrently.")
    parser.add_argument("--tool-choice", choices=["required", "auto"], default="auto",
                        help="'auto' (default) leaves tool_choice unset and relies on the prompt; "
                             "'required' forces a tool call.")

    args = parser.parse_args()

    timeout = httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT)
    if args.base_url is not None:
        print(f"I: base-url override: {args.base_url}")
        if args.api_key is None:
            # To avoid leaking API KEY
            print(f"W: base-url override: won't use OPENAI_API_KEY env var, using 'sk-no-key' as API key. Pass --api-key if needed.")
            api_key = "sk-no-key"
        else:
            api_key = args.api_key
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key, timeout=timeout, max_retries=0)
    else:
        client = openai.OpenAI(api_key=args.api_key, timeout=timeout, max_retries=0)

    print(f"I: model: {args.model}")
    print(f"I: reasoning_effort: {args.reasoning_effort!r}")

    if args.samples_file is not None:
        with open(args.samples_file, 'r') as f:
            batches = [s.strip() for s in f.readlines() if s.strip()]
    elif args.samples is not None:
        batches = [args.samples]
    else:
        print(f"E: either samples or samples-file must be provided")
        exit(1)

    model_name = args.alias if args.alias is not None else args.model

    def process(i, batch):
        tag = f"[{i + 1}/{len(batches)}]"
        samples = []
        paths = batch.split(",")
        for p in paths:
            with open(p, 'r') as file:
                samples.append(json.load(file))
                print(f"I: {tag} {p} : {samples[-1]['secret']}")

        res = run(client, samples, args)

        content = {
            "paths" : paths,
            "args" : {
                "model": model_name,
                "reasoning_effort": args.reasoning_effort,
                "n_tokens_max": args.n_tokens_max,
            },
            "result" : res,
        }
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(f"logs/{dt}-{secrets.token_hex(3)}.json", "x") as fw:
            fw.write(json.dumps(content))

        print(f"Result: {tag} {res}")

    # Batches are independent API calls, so run up to --jobs of them at once.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process, i, batch) for i, batch in enumerate(batches)]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
