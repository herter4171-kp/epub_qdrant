I checked the official Qwen guidance: use normal `system`/`user`/`assistant` chat structure, control thinking mode explicitly where supported, avoid greedy decoding, and use non-thinking/instruct mode when you need structured deterministic implementation behavior rather than visible reasoning. ([Qwen][1])

````markdown
# Instructions for Qwen3.6-35B: Harvest LLM Responses from Query Results

You are modifying an existing codebase. Stay tightly scoped to the task described here. Do not redesign the experiment, scoring pipeline, statistical analysis, evaluation model, dashboard, schema, or future reporting. The only goal is to add a response-harvesting pipeline that reads existing query result JSON files and writes corresponding LLM response JSON files.

Use careful implementation judgment, but do not expand the product scope.

## Model Prompting Guidance for Qwen

Use Qwen in a normal chat-message format:

1. `system` message first.
2. `assistant` message second.
3. `user` message third.

Do not concatenate everything into one user prompt unless the existing codebase already makes that unavoidable.

For this task, prefer deterministic, implementation-oriented behavior. If the runtime supports Qwen thinking controls, use non-thinking mode for the actual production response calls unless the existing local setup explicitly requires otherwise.

Recommended generation posture for response harvesting:

- non-thinking / instruct mode if available
- no visible `<think>` content in saved outputs
- stable generation settings
- no greedy decoding if the serving stack warns against it
- preserve raw model response text exactly as returned, except for normal transport-level decoding

The response-harvesting code should not depend on hidden reasoning text.

## Existing Input Concept

There is an existing directory named:

```text
query_results
````

Each file in `query_results` is a JSON result file. Each result file contains one benchmark prompt case.

A result file has fields similar to:

```json
{
  "id": 1,
  "category": "spatial_orientation",
  "proficiency": 1,
  "prompt": "Can you show me the paragraph that comes right after this one in a document?",
  "topk": 2,
  "timestamp": "2026-04-25T19:13:17.502358+00:00",
  "sources": {
    "papers": { ... },
    "papers_semantic": { ... },
    "bedrock": { ... }
  }
}
```

The exact internals of each source may differ. Do not normalize, reinterpret, summarize, or modify the source payload before inserting it into the LLM message.

The `papers_semantic` source is expected to have `2 * topk` results. This is intentional. Do not “fix” it, warn about it, or force it to match the outer `topk`.

## Output Concept

Create a new directory at the same level as `query_results`:

```text
query_responses
```

For each processed input file:

```text
query_results/<filename>.json
```

write a corresponding output file:

```text
query_responses/<filename>.json
```

The output filename must be identical to the input filename.

Do not invent a new naming convention.

Do not append method names to filenames.

Do not split one input file into multiple output files.

## Processing Unit

For each input result JSON file:

1. Read the JSON.
2. Extract the fixed prompt from the result file’s top-level `prompt` field.
3. Iterate over each entry in the top-level `sources` object.
4. For each source, make one LLM call.
5. Save all source responses for that input file into the corresponding output JSON file.

If an input file has three sources:

```text
papers
papers_semantic
bedrock
```

then the output file should contain three harvested responses, one per source.

## Required LLM Message Shape

For each source in a result file, construct messages exactly in this conceptual shape:

```json
[
  {
    "role": "system",
    "content": "<TBD_SYSTEM_PROMPT>"
  },
  {
    "role": "assistant",
    "content": "Results: <RAW_TOPK_OUTPUT_FOR_THIS_SOURCE>"
  },
  {
    "role": "user",
    "content": "<PROMPT_FROM_RESULT_FILE>"
  }
]
```

Important details:

* The system prompt is TBD and should be configurable.
* The assistant message must begin with the literal header:

```text
Results: 
```

* After `Results: `, include the raw top-k output for that source.
* The raw source output should be serialized in a stable way if it is not already a string.
* Do not summarize the source.
* Do not flatten the source.
* Do not remove metadata.
* Do not rewrite chunks.
* Do not transform Bedrock output into the same schema as the other sources.
* Do not alter the user prompt.
* Do not add extra instructions into the user message.
* Do not append scoring rubrics.
* Do not ask the LLM to evaluate anything unless that is already present in the TBD system prompt.

The reason for using an assistant message as source-of-truth context is already understood and intentional. Do not challenge or change this pattern.

## System Prompt Handling

Add a way to provide the TBD system prompt.

Acceptable approaches, in order of preference:

1. CLI flag pointing to a system prompt file.
2. CLI flag containing the system prompt text directly.
3. Existing project configuration mechanism, if the codebase already has one.

The implementation should make the system prompt explicit and easy to swap.

Example CLI shape:

```bash
python harvest_responses.py \
  --query-results-dir query_results \
  --query-responses-dir query_responses \
  --system-prompt-file prompts/response_system.md
```

If the codebase already has a CLI style, follow the existing style instead of forcing this exact interface.

## Limit Flag

The pipeline must support a limit flag.

The limit flag is an integer number of input result files to process.

Example:

```bash
python harvest_responses.py --limit 10
```

Behavior:

* Determine the standard file ordering.
* Process only the first `N` input files from that ordering.
* The limit applies to result files, not source calls.
* If `--limit 10` and each result file has 3 sources, up to 30 LLM calls may occur.
* If `--limit` is omitted, process all input files.
* If `--limit 0`, process zero files and exit successfully.
* If `--limit` is negative, fail with a clear error.

Use stable standard file ordering. Prefer lexicographic filename ordering unless the existing codebase has an established ordering convention.

## Idempotency and Existing Outputs

Default behavior should be safe.

If the corresponding output file already exists, prefer skipping it unless the codebase already has overwrite semantics.

Add an explicit overwrite flag if appropriate:

```bash
--overwrite
```

Expected behavior:

* Without `--overwrite`: skip files that already have a response file.
* With `--overwrite`: regenerate and replace the output file.

Do not partially append to an existing output file unless the existing architecture strongly favors append-only logs.

## Output JSON Shape

The output JSON should preserve enough information to trace every harvested response back to its input.

Recommended shape:

```json
{
  "input_file": "example.json",
  "input_id": 1,
  "category": "spatial_orientation",
  "proficiency": 1,
  "topk": 2,
  "prompt": "Can you show me the paragraph that comes right after this one in a document?",
  "timestamp": "2026-04-25T19:13:17.502358+00:00",
  "responses": {
    "papers": {
      "messages": [
        {
          "role": "system",
          "content": "<TBD_SYSTEM_PROMPT>"
        },
        {
          "role": "assistant",
          "content": "Results: <RAW_TOPK_OUTPUT_FOR_THIS_SOURCE>"
        },
        {
          "role": "user",
          "content": "<PROMPT_FROM_RESULT_FILE>"
        }
      ],
      "response_text": "<RAW_MODEL_RESPONSE_TEXT>",
      "model": "<MODEL_NAME>",
      "started_at": "<ISO_TIMESTAMP>",
      "completed_at": "<ISO_TIMESTAMP>",
      "elapsed_seconds": 0.0,
      "error": null
    },
    "papers_semantic": {
      "messages": [],
      "response_text": "",
      "model": "<MODEL_NAME>",
      "started_at": "<ISO_TIMESTAMP>",
      "completed_at": "<ISO_TIMESTAMP>",
      "elapsed_seconds": 0.0,
      "error": null
    },
    "bedrock": {
      "messages": [],
      "response_text": "",
      "model": "<MODEL_NAME>",
      "started_at": "<ISO_TIMESTAMP>",
      "completed_at": "<ISO_TIMESTAMP>",
      "elapsed_seconds": 0.0,
      "error": null
    }
  }
}
```

This shape is a recommendation, not a license to overbuild. If the existing project already has an output convention, preserve it while ensuring the following are stored:

* input filename
* original prompt
* source name
* exact messages sent to the model
* raw response text
* model identifier
* error information if the call failed

## Raw Source Serialization

For the assistant message, the content must be:

```text
Results: <serialized source payload>
```

If the source payload is a string, use it as-is.

If the source payload is JSON/object/list data, serialize it as pretty JSON with stable key ordering if possible.

Recommended:

```python
json.dumps(source_payload, ensure_ascii=False, sort_keys=True, indent=2)
```

Then:

```python
assistant_content = "Results: " + serialized_source_payload
```

Do not add Markdown fences around the serialized source unless the existing prompting code already does that.

## Error Handling

The pipeline should continue processing other files when one source call fails, unless the existing project convention is fail-fast.

For each failed source response, write an error record that includes:

* source name
* exception type
* exception message
* timestamp
* whether the call was attempted
* messages prepared for the call, if safe and useful

Do not silently drop failed sources.

Do not create a response file that looks successful if every source failed.

## Logging

Log progress at file granularity and source granularity.

Example:

```text
Processing 0001.json
  calling source: papers
  calling source: papers_semantic
  calling source: bedrock
Wrote query_responses/0001.json
```

For `--limit`, log the number of selected files:

```text
Selected 10 input files from query_results
```

For skipped files:

```text
Skipping 0001.json because query_responses/0001.json already exists
```

## CLI Requirements

Implement or update a CLI entry point with these minimum capabilities:

```text
--query-results-dir
--query-responses-dir
--system-prompt-file or equivalent
--limit
--overwrite
--model / model identifier if the existing codebase supports it
```

Defaults may be:

```text
--query-results-dir query_results
--query-responses-dir query_responses
```

Do not require absolute paths.

## Directory Requirements

When the pipeline starts:

1. Confirm `query_results` exists.
2. Create `query_responses` if it does not exist.
3. Fail clearly if the input path is not a directory.
4. Fail clearly if no JSON files are found, unless `--limit 0`.

The new `query_responses` directory should be a sibling of `query_results` by default.

Example:

```text
project_root/
  query_results/
    0001.json
    0002.json
  query_responses/
    0001.json
    0002.json
```

## Ordering Requirement

Use stable file ordering.

Recommended:

```python
files = sorted(query_results_dir.glob("*.json"))
```

Do not use filesystem iteration order directly.

The limit must be applied after sorting:

```python
selected_files = files[:limit]
```

## Do Not Implement These Yet

Do not implement scoring.

Do not implement judging.

Do not implement dashboards.

Do not implement CUDA analysis.

Do not implement statistical modeling.

Do not implement prompt category analysis.

Do not normalize source schemas.

Do not validate whether `papers_semantic` returns `2 * topk`.

Do not change retrieval.

Do not rerun retrieval.

Do not modify files under `query_results`.

Do not design a database.

Do not add Parquet export.

Do not add Qdrant ingestion.

Do not add answer comparison logic.

Do not add pairwise method comparisons.

This task is only response harvesting.

## Acceptance Criteria

The implementation is complete when:

1. Running the script creates `query_responses` beside `query_results`.
2. Each processed input JSON produces one output JSON with the same filename.
3. Each source in the input JSON produces one LLM call.
4. Each LLM call uses exactly three conceptual messages:

   * system: TBD/configured prompt
   * assistant: `Results: ` plus raw source payload
   * user: top-level prompt from the result JSON
5. The `--limit` flag processes only the first N files by stable file ordering.
6. Existing outputs are skipped unless overwrite is explicitly enabled.
7. The raw messages sent to the model are saved with the response.
8. Failed calls are recorded rather than silently discarded.
9. No evaluation, scoring, analysis, or visualization is added.

## Minimal Pseudocode

```python
def load_system_prompt(args):
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8")
    if args.system_prompt:
        return args.system_prompt
    raise ValueError("A system prompt must be provided")


def serialize_source_payload(payload):
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def build_messages(system_prompt, source_payload, prompt):
    serialized = serialize_source_payload(source_payload)
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "assistant",
            "content": "Results: " + serialized,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def select_input_files(query_results_dir, limit):
    files = sorted(Path(query_results_dir).glob("*.json"))
    if limit is None:
        return files
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    return files[:limit]


def process_file(input_path, output_path, system_prompt, llm_client):
    result = json.loads(input_path.read_text(encoding="utf-8"))

    prompt = result["prompt"]
    sources = result.get("sources", {})

    output = {
        "input_file": input_path.name,
        "input_id": result.get("id"),
        "category": result.get("category"),
        "proficiency": result.get("proficiency"),
        "topk": result.get("topk"),
        "prompt": prompt,
        "timestamp": result.get("timestamp"),
        "responses": {},
    }

    for source_name, source_payload in sources.items():
        messages = build_messages(system_prompt, source_payload, prompt)
        started_at = now_iso()

        try:
            response = llm_client.chat(messages=messages)
            completed_at = now_iso()

            output["responses"][source_name] = {
                "messages": messages,
                "response_text": response.text,
                "model": response.model,
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_seconds": elapsed_seconds(started_at, completed_at),
                "error": None,
            }

        except Exception as exc:
            completed_at = now_iso()

            output["responses"][source_name] = {
                "messages": messages,
                "response_text": None,
                "model": getattr(llm_client, "model", None),
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_seconds": elapsed_seconds(started_at, completed_at),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
```

## Questions to Resolve Only If the Codebase Does Not Already Answer Them

Do not stop implementation for these unless the codebase provides no obvious path.

1. What is the existing LLM client wrapper?

   * OpenAI-compatible client?
   * llama.cpp endpoint?
   * vLLM endpoint?
   * custom local wrapper?

2. Where should the TBD system prompt live?

   * CLI file path?
   * existing config file?
   * environment variable?
   * checked-in prompt file?

3. What is the current standard filename ordering?

   * lexicographic filename order is acceptable if no existing convention exists.

4. Should existing response files be skipped by default?

   * implement skip-by-default plus `--overwrite` unless the project already says otherwise.

5. Should failed source calls still produce an output file?

   * yes, unless every existing pipeline component uses fail-fast semantics.

6. Does the existing codebase already have timestamp helpers, logging conventions, or model-call wrappers?

   * use them if they exist.

## Implementation Style

Prefer small functions.

Prefer boring, inspectable code.

Avoid clever abstractions.

Keep the core pipeline readable:

```text
read files
sort files
apply limit
for each file:
  read JSON
  for each source:
    build messages
    call model
    store response
  write output JSON
```

This is a harvesting step, not an analysis framework.

```
::contentReference[oaicite:1]{index=1}
```

[1]: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html "Quickstart - Qwen"

