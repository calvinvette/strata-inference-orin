# Jetson AGX Orin validation, 2026-10-08

Hardware: Jetson AGX Orin Developer Kit 32 GB, ARM64, JetPack 6.2,
CUDA 12.6.68, GCC 11.4, SM87. Power mode is MAXN; clocks were not locked.
The user stopped their separate ninfer service before these runs.
Source baseline: `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`, with the
uncommitted `feat/jetson-orin-arm64` changes. Installed engine version
0.1.40.3, engine source fingerprint `71ddb783ff5dddc8`, vision fingerprint
`bccffef4afd95604`; [build metadata](BUILD.json) preserves these values.

Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, IQ1_M,
revision `5348543e0147355ac9cbcb031184a3546350988e`.
Both shards were checked against their pinned SHA256 values. The expert pack
is file-backed (23.42 GiB); canonical dense weights occupy 1.37 GiB.
MTP uses all 31 verified tensors from revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, prepared as Q2_0 experts.

## Findings and fixes

A one-token native CLI prompt originally exited successfully without producing
a token because position zero was also the sentinel for skipping execution.
Using -1 as the sentinel fixes this. [Before](single-token-before-fix.txt)
and [after](single-token-after-fix.txt) runs preserve the evidence.

The first MTP API request originally reduced available RAM below the default
six GiB headroom. Auto cache sizing now leaves a separate three GiB reserve
for late prompt/verification/CUDA workspace on integrated devices, and startup
checks available physical RAM after initialization. Explicit reserve settings
remain user-controlled. This is a sizing heuristic and a point-in-time check;
it cannot prevent another process from allocating RAM afterwards.

The corrected API server allocated 13.39 GiB to 7,036 cached experts.
[Seven API checks](api-validation-workspace.json) passed: repeated greedy
requests, OpenAI and Anthropic streaming, and recovery after cancellation
during decode and prefill. All listeners used `127.0.0.1:18081`.

## Repeated throughput

[Raw responses](benchmark-4k.json) and [timings](benchmark-4k.txt) record
three requests at each prompt size, with 64 generated tokens each, MTP spec 4,
prefill batch 64 and context limit 4096. No prompt tokens were reused.

| Prompt tokens | Median prompt tokens/s | Median decode tokens/s |
| --- | --- | --- |
| 172 | 37.8 | 21.6 |
| 557 | 47.9 | 19.7 |
| 1069 | 52.5 | 18.6 |

Across the corrected API checks, benchmarks and 3105-token context request,
[one-second samples](api-workspace-samples.jsonl) show at least 7.43 GiB
available physical RAM. Swap usage reached 1.58 GiB; this was not a swap-free
run. RSS alone omits GPU allocations, so the report uses system available RAM.
The [3105-token prompt](context-4k.json) returned the expected `OK` with
56.2 prompt tokens/s; its two-token decode is not a throughput benchmark.

## CPU reference

For the explicit input token 248045, the actual CPU llama.cpp reference and
Strata both selected token 846. All 248,320 logits were finite.
[Recorded differences](first-logit-comparison.json) include RMS logit error
0.01759, selected-token log probability difference 0.00000893 and
KL(reference || Strata) 8.39e-8. This tests one position, not all prompts.
Use the pinned [reference helper](../2026-10-07-jetson-orin/reference_logits.cpp)
and `STRATA_DUMP_FIRST_LOGITS=OUTPUT` with Strata `--tokens`, `--max-new 1`,
`--prefill 1`. Compare the last headered reference row with the raw FP32
Strata row. Native packs do not implement `--dump-logits`.

The [7234-token prompt](context-8k.json) passed at context limit 8192,
returning `OK` at 49.7 prompt tokens/s. [Samples](memory-8k.jsonl) include
startup and this request.

The [15 Jetson setup cases](setup-jetson.txt) and
[seven built ARM CPU tests](cpu-selected-tests.txt) passed. An unrestricted
[CTest invocation](cpu-tests.txt) was unsuitable for this partial build:
15 target executables were absent and the expert parity target expected a
model file at `pack/full/experts.bin`. Its failure is not counted as a passing
suite. Earlier explicit real-expert parity results remain in the October 7
report. The [desktop setup regression suite](setup-desktop.txt) also passed 110 cases.

The installed configurations are preserved alongside the results. Start the
MTP server with `.venv/bin/python -m serve.server --engine strata --config`
followed by the absolute path to `strata-orin-validation-mtp.json`, and
`--host 127.0.0.1 --port 18081`. Run `validate_api.py --out OUTPUT` then
`benchmark_api.py --out OUTPUT`. Use `monitor_memory.py --config CONFIG
--out OUTPUT` while the server runs. Paths in the configurations identify the
verified prepared artifacts; substitute equivalent paths on another host.

The [31234-token prompt](context-32k.json) passed at context limit 32768,
returning `OK` after 584 seconds of prefill (53.5 tokens/s). A subsequent
[short request](recovery-32k.json) also passed. The 32K cache contained 6598
experts (12.55 GiB). [Memory samples](memory-32k.jsonl) cover startup, the long
prompt and recovery: minimum available RAM 7.33 GiB, maximum system RAM in use
22.65 GiB, maximum process-tree RSS 9.28 GiB, maximum swap in use 0.99 GiB.
Host-wide physical disk reads increased by 26.82 GiB over this window; this
includes startup and unrelated host activity. Engine expert-read counters are
logical reads and do not measure physical storage traffic.
At context 8192, the corresponding sampled minimum was 7.48 GiB available
RAM and host-wide reads increased by 18.65 GiB across startup and the request.

GPU temperatures during the 32K window ranged from 48.3 to 57.8 degrees C
in [one-second tegrastats samples](tegrastats-32k.txt). These temperatures
do not establish sustained throughput after thermal equilibrium.

For [two- and four-token inputs](reference-comparison.json), the CPU reference
and both Strata prompt paths selected token 248046, but log probabilities
were substantially different: KL(reference || Strata) was 0.356–1.118.
For four tokens the selected-token log probability differed by 0.983–1.005.
These are observations, not numerical-parity passes. For the model's
[formatted 19-token chat prompt](reference-chat-prompt.json),
[both prompt paths](reference-chat-comparison.json) selected token 9419
(`Hello`), matching the CPU reference. The selected-token log probability
differed by 0.00036–0.00038 and KL was 3.45e-5–3.86e-5. Logit RMS differences
were 0.158–0.168. This establishes agreement for the tested prompt, not a
universal numerical tolerance or parity for every sequence. The raw-token
discrepancy remains documented and has not been attributed to ARM versus
existing engine approximations without a desktop reference run.

## Vision

The verified BF16 projector encoded the same 56-by-56 fixture on CPU and CUDA.
[Both outputs](vision-comparison.json) had 16 tokens, a 4-by-4 grid and width
2560, with all values finite. Cosine similarity was 0.99989; relative RMS
error was 0.01484. Reported encoding times were 6964 ms on CPU and 28 ms on
CUDA. Whole-process times including startup/warmup were 8.69 and 3.48 seconds.
These measurements concern this fixture and a 16-token encoder cap.

The [image API request](image-api.json) generated 48 tokens describing the
colored checkerboard and gradient. The [repeated request](image-api-repeat.json)
returned identical text with 34 cached prompt tokens. The encoder loaded before
cache sizing; the cache held 6428 experts (12.22 GiB). Across startup and both
requests, [memory samples](memory-vision.jsonl) retained at least 7.38 GiB
available RAM, with peak system use 22.60 GiB and swap use 1.29 GiB.
This does not validate high-resolution images or the default encoder token cap.
Run `validate_vision.py` with the prepared paths, then use the saved
`strata-orin-validation-vision.json` server configuration and
`validate_image_api.py`. The validation server was stopped after the tests.

## Small performance comparisons

[The CPU row benchmark](cpu_rows.cpp) compares portable Q2 rows and pinned
ggml's ARM backend over 256 rows, width 2560 and three tokens, with 20 timed
iterations after warmup. [Mean times](cpu-rows.txt) were 2.303 ms and 0.323 ms.
They use different Q8_1/Q8_0 activation contracts; outputs were finite with
relative RMS difference 0.00020. This is not an end-to-end expert benchmark or
permission to substitute activation contracts. Existing CPU correctness tests
check each contract independently. Build with the pinned ggml include directory,
`libstrata_kernels_cpu.a`, `libggml-cpu.a`, `libggml-base.a`, OpenMP and pthread.

[The memory benchmark](memory_paths.cu) fills a 64 MiB buffer on the CPU,
then measures upload (where needed), GPU checksum and synchronization. CPU fill
is excluded. After three warmups, ten iterations produced these
[median times](memory-paths.txt), all with exact checksums:

| Sequential path | Median ms |
| --- | --- |
| Pinned host copy to device | 13.169 |
| Mapped host GPU read | 5.345 |
| Managed allocation | 5.754 |

Compile with `/usr/local/cuda-12/bin/nvcc -O2 -arch=sm_87` and run with the
model stopped. The managed test transfers ownership only after device
synchronization. Orin lacks concurrent managed access; these timings do not
justify managed allocations for concurrently polled control buffers, or replacing
the expert-cache policy. No new custom NEON or managed-memory optimization was
enabled from these microbenchmarks.

## Tested scope and limits

The port is validated for the pinned Coder IQ1_M pack on this Orin, with
file-backed experts, auto cache, prefill 64, MTP spec 4, context limits up to
32768 and six GiB physical headroom plus three GiB late-workspace reserve.
Model preparation and local source builds preserve JetPack's CUDA libraries.
Windows, HIP and other model variants were not run on this host. Desktop setup
regressions and x86 CPU cross-builds passed; this is not desktop GPU validation.
The raw-token numerical differences above remain a limit on parity claims.
No process-independent RAM reservation or sustained thermal benchmark is claimed.
The validation processes are stopped; the user's ninfer service remains inactive.
