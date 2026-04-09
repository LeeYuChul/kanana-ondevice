# Kanana On-Device Converter

`kakaocorp/kanana-nano-2.1b-base` → MediaPipe `.task` 변환 파이프라인

---

## 프로젝트 구조

```
ondevice/
├── scripts/
│   ├── export_kanana_mediapipe.py   # STEP 1: HuggingFace → multi-sig TFLite
│   ├── bundle_task_fixed.py         # STEP 2: TFLite → .task 패키징
│   ├── convert_tokenizer.py         # (보조) tokenizer.json → SentencePiece (수동)
│   ├── inspect_model.py             # TFLite 서명 구조 확인
│   ├── verify_artifact.py           # STEP 3: .task 검증
│   └── outputs/
│       ├── kanana_nano_2_1b_q4_block128_ekv1280.tflite   # 1035 MB
│       └── kanana_nano_2_1b_int4.task                    # 1037 MB
├── tokenizers/
│   └── kanana_official_tokenizer.model   # SentencePiece protobuf (2.2 MB)
└── README.md
```

---

## 환경

```bash
# Python 가상환경 (~/문서/Study/kanana-test/venv)
cd ~/문서/Study/kanana-test
source venv/bin/activate

# 주요 패키지
# litert_torch == 0.8.0
# torch == 2.9.0+cu128
# transformers, sentencepiece, huggingface_hub
```

---

## 변환 파이프라인

### STEP 1: TFLite 변환 (`export_kanana_mediapipe.py`)

MediaPipe LlmInference가 요구하는 **multi-signature TFLite**를 생성합니다.
- 출력 서명: `prefill_256` + `decode`
- 단일 `serving_default` 서명은 **사용 불가** ("Decode runner not found" 오류)

```bash
cd ~/문서/Study/kanana-test
source venv/bin/activate

python ondevice/scripts/export_kanana_mediapipe.py \
  --checkpoint kakaocorp/kanana-nano-2.1b-base \
  --output_path ondevice/scripts/outputs \
  --output_name kanana_nano_2_1b \
  --quantize dynamic_int4_block128 \
  --prefill_seq_lens 256 \
  --kv_cache_max_len 1280
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | `kakaocorp/kanana-nano-2.1b-base` | HuggingFace 모델 ID 또는 로컬 경로 |
| `--quantize` | `dynamic_int4_block128` | `none` / `dynamic_int8` / `dynamic_int4_block32` / `dynamic_int4_block128` |
| `--prefill_seq_lens` | `256` | 공백 구분 다중 입력 가능 (예: `128 256 512`) |
| `--kv_cache_max_len` | `1280` | KV 캐시 최대 길이 |

출력: `outputs/kanana_nano_2_1b_q4_block128_ekv1280.tflite` (1035 MB)

---

### STEP 1.5: 토크나이저 변환 (litert_torch 공식 도구)

HuggingFace tokenizer → SentencePiece protobuf 변환.  
**반드시 이 방법을 사용할 것** — `tokenizer.json` 직접 번들 시 "ParseFromString failed" 오류.

```bash
KANANA_SNAPSHOT=$(ls ~/.cache/huggingface/hub/models--kakaocorp--kanana-nano-2.1b-base/snapshots/)

python venv/lib/python3.12/site-packages/litert_torch/generative/tools/tokenizer_to_sentencepiece.py \
  --checkpoint ~/.cache/huggingface/hub/models--kakaocorp--kanana-nano-2.1b-base/snapshots/${KANANA_SNAPSHOT} \
  --output_path ondevice/tokenizers/kanana_official_tokenizer.model \
  --normalize_tokens decode
```

검증 결과 (정상):
```
String to verify: Hello, world! How are you?   → PASS
String to verify: Instruct: write a python...   → PASS
Not matched strictly ~4%,  loosely ~1% (허용 범위 내)
```

출력: `tokenizers/kanana_official_tokenizer.model` (2.2 MB)

---

### STEP 2: `.task` 번들 생성 (`bundle_task_fixed.py`)

```bash
python ondevice/scripts/bundle_task_fixed.py --quant int4
```

#### 번들 구조 (`kanana_nano_2_1b_int4.task`)

| 항목 | 파일 | 크기 |
|------|------|------|
| `TF_LITE_PREFILL_DECODE` | TFLite (multi-sig) | 1035 MB |
| `TOKENIZER_MODEL` | SentencePiece protobuf | 2.2 MB |
| `METADATA` | JSON | < 1 KB |

**METADATA 내용:**
```json
{
  "model_type": "LLM_CAUSAL",
  "model_id": "kanana-nano-2.1b-base",
  "quant_mode": "int4_block128",
  "start_token_id": 128000,
  "stop_token_ids": [128001],
  "vocab_size": 128256,
  "max_seq_len": 8192
}
```

> ⚠️ **ZIP 압축 방식**: 반드시 `ZIP_STORED` (비압축) 사용  
> `ZIP_DEFLATED` 사용 시 → "Expected uncompressed zip archive" 오류

---

### STEP 3: 번들 검증 (`verify_artifact.py`)

```bash
python ondevice/scripts/verify_artifact.py --quant int4
```

---

## 산출물

| 파일 | 크기 | 설명 |
|------|------|------|
| `outputs/kanana_nano_2_1b_q4_block128_ekv1280.tflite` | 1035 MB | multi-sig TFLite (prefill_256 + decode) |
| `outputs/kanana_nano_2_1b_int4.task` | 1037 MB | MediaPipe .task 번들 |
| `tokenizers/kanana_official_tokenizer.model` | 2.2 MB | SentencePiece protobuf |

---

## Flutter 앱 연결

```bash
# 로컬 Flutter 프로젝트로 복사 (Mac에서)
scp mjudcd@192.168.0.4:~/문서/Study/kanana-test/ondevice/scripts/outputs/kanana_nano_2_1b_int4.task \
    /path/to/hellomaple_test/assets/models/
```

Flutter 앱에서의 모델 등록 (`lib/models/model.dart`):
```dart
kanana_nano_2_1b_int4(
  baseUrl: 'assets/models/kanana_nano_2_1b_int4.task',
  filename: 'kanana_nano_2_1b_int4.task',
  displayName: 'Kanana Nano 2.1B Int4 (Local)',
  preferredBackend: Backend.cpu,
  localModel: true,
)
```

---

## 모델 아키텍처 (Kanana-Nano-2.1B-Base)

| 항목 | 값 |
|------|-----|
| 아키텍처 | LlamaForCausalLM |
| vocab_size | 128256 |
| hidden_size | 1792 |
| num_hidden_layers | 32 |
| num_attention_heads | 24 |
| num_key_value_heads | 8 (GQA) |
| intermediate_size | 8064 |
| rope_theta | 500000 |
| max_position_embeddings | 8192 |
| tie_word_embeddings | True |
| BOS token | `<\|begin_of_text\|>` (id=128000) |
| EOS token | `<\|end_of_text\|>` (id=128001) |

---

## 알려진 오류 및 해결책

| 오류 | 원인 | 해결 |
|------|------|------|
| `Decode runner not found` | TFLite에 `decode` 서명 없음 | `export_kanana_mediapipe.py` 사용 (multi-sig 변환) |
| `Expected uncompressed zip archive` | ZIP_DEFLATED 압축 | `zipfile.ZIP_STORED` 사용 |
| `sentencepiece ParseFromString failed` | TOKENIZER_MODEL이 JSON | SentencePiece protobuf 변환 후 번들 |
| `spm.bos_id()` 반환 -1 | SP Python API 특성 | 무시해도 됨 — MediaPipe는 `trainer_spec.bos_id` 직접 읽음 |
