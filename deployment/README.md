# Deploying to Ollama

This turns your trained LoRA adapter into a standalone model you can
run with `ollama run truck-dispatch-ai`, just like any other local
chat model — no Python environment, no `transformers`, no reference to
this repo needed at runtime.

The pipeline has three steps:

```
LoRA adapter (training/output/)
        │  merge_adapter.py
        ▼
Merged HF model (models/merged/)
        │  convert_to_gguf.ps1 / .sh
        ▼
fp16 GGUF (models/gguf/truck-dispatch-f16.gguf)
        │  (optional) llama-quantize
        ▼
Quantized GGUF (models/gguf/truck-dispatch-q4_k_m.gguf)
        │  ollama create -f deployment/Modelfile
        ▼
ollama run truck-dispatch-ai
```

## 1. Merge the adapter into the base model

```bash
python training/merge_adapter.py
```

This loads the base model in fp16, folds the trained LoRA weights into
it, and saves a full standalone model to `models/merged/`. This step
needs CPU RAM roughly equal to the model's fp16 size (a 4B model is
~8 GB) — a GPU is not required here, just enough system memory.

## 2. Convert to GGUF

Install [Ollama](https://ollama.com/download) if you haven't already,
then convert the merged model:

**Windows (PowerShell):**
```powershell
.\deployment\convert_to_gguf.ps1
```

**macOS / Linux:**
```bash
bash deployment/convert_to_gguf.sh
```

Both scripts clone `llama.cpp` (for its Python conversion script only
— no C++ build required) and produce
`models/gguf/truck-dispatch-f16.gguf`. This fp16 file is already usable
with Ollama as-is; quantizing further is optional but recommended for
faster loading and a smaller file.

### Optional: quantize for a smaller, faster model

Download a prebuilt `llama.cpp` release for your OS from
[the releases page](https://github.com/ggml-org/llama.cpp/releases)
(look for a `llama-quantize` or `quantize` binary — no compiling
needed), then run:

```bash
llama-quantize models/gguf/truck-dispatch-f16.gguf models/gguf/truck-dispatch-q4_k_m.gguf Q4_K_M
```

`Q4_K_M` is a good default: roughly a 60–70% size reduction with
minimal quality loss. `Q5_K_M` or `Q8_0` trade some of that size saving
back for higher fidelity if you have the disk space and RAM to spare.

## 3. Build and run the Ollama model

`deployment/Modelfile` already points at the quantized file. If you
skipped quantization, edit the `FROM` line to point at the fp16 file
instead.

```bash
ollama create truck-dispatch-ai -f deployment/Modelfile
ollama run truck-dispatch-ai
```

From here it behaves like any other Ollama model — you can query it
via `ollama run`, the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)
(`POST http://localhost:11434/api/generate`), or any tool that talks to
Ollama.

## Switching personas

The dataset covers two system prompts — general business
correspondence and cold outreach. `Modelfile` ships with the
cold-outreach persona as the default `SYSTEM` prompt. To use the other
persona (or a custom one) for a session without rebuilding:

```bash
ollama run truck-dispatch-ai
>>> /set system You are a professional business correspondence assistant for a truck dispatching service. Write clear, courteous, and fully professional emails and responses. Use correct grammar and punctuation at all times. Do not make unsupported claims. Do not use aggressive sales language, slang, or casual phrasing.
```

Or pass a `"system"` field in the API request instead of editing the
Modelfile.

## Troubleshooting

- **Responses look malformed / include stray tokens like `<|im_start|>`
  in the output.** Uncomment the `TEMPLATE` block in `Modelfile` (it's
  commented out by default because Ollama usually picks up the correct
  chat template automatically from the GGUF metadata), then re-run
  `ollama create`.
- **`convert_hf_to_gguf.py` fails on an unrecognized architecture.**
  Make sure `models/merged/` contains a `config.json` with the correct
  `model_type`, and that your `llama.cpp` clone is reasonably recent —
  delete `deployment/llama.cpp` and re-run the conversion script to
  pull a fresh copy.
- **Out of memory during merge or conversion.** Both steps run on CPU
  and need RAM roughly equal to the model's fp16 size. Close other
  applications, or run on a machine with more system RAM — a GPU is
  not used for either step.
