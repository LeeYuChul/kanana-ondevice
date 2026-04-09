# On-device conversion pipeline for Kanana

이 디렉터리는 `kakaocorp/kanana-nano-2.1b-base`를 온디바이스용 산출물(`.task`, `.litertlm`)로 변환하기 위한 실험 파이프라인입니다.

## 목표
1. 사전 점검(모델/토크나이저/특수 토큰/아키텍처)을 자동화
2. 8-bit, 4-bit 양자화 내보내기 시도
3. `.task` / `.litertlm` 번들링 메타데이터 생성
4. 최종 산출물 검증(파일/토큰/샘플 추론)

## 구조
- `configs/`: 변환 파라미터 YAML
- `scripts/inspect_model.py`: 아키텍처/토크나이저/토큰/호환성 점검
- `scripts/export_8bit.py`: 8bit 양자화 변환 시도
- `scripts/export_4bit.py`: 4bit 양자화 변환 시도
- `scripts/bundle_task.py`: `.task` 또는 `.litertlm` 번들 생성
- `scripts/verify_artifact.py`: 번들 내용/샘플 실행 검증
- `outputs/`: 결과물 저장
- `logs/`: 로그 저장

## 빠른 시작
```bash
python ondevice/scripts/inspect_model.py \
  --model-id kakaocorp/kanana-nano-2.1b-base \
  --out ondevice/logs/inspect_report.json

python ondevice/scripts/export_8bit.py \
  --config ondevice/configs/convert_8bit.yaml

python ondevice/scripts/export_4bit.py \
  --config ondevice/configs/convert_4bit.yaml

python ondevice/scripts/bundle_task.py \
  --input-dir ondevice/outputs/int8 \
  --format task \
  --output ondevice/outputs/int8/kanana-int8.task

python ondevice/scripts/verify_artifact.py \
  --artifact ondevice/outputs/int8/kanana-int8.task \
  --sample-prompt "안녕하세요. 자기소개 해주세요."
```

## 의존성
권장:
- `transformers>=4.40`
- `torch`
- `accelerate`
- `bitsandbytes` (4/8bit 양자화)
- `pyyaml`

옵션:
- LiteRT / AI Edge Torch 관련 Python 패키지(환경별 상이)

> 참고: Kanana 계열의 직접 `.task` / `.litertlm` 변환은 환경/도구 버전에 따라 실패할 수 있습니다. 이 스크립트들은 **사전 점검 + 재현 가능한 시도 경로**를 제공하는 데 초점을 둡니다.

## Troubleshooting
- `StrictDataclassClassValidationError` (예: `hidden size ... is not a multiple ...`)가 `inspect_model.py`에서 발생하던 케이스를 피하기 위해, 현재 inspect 스크립트는 `AutoConfig` 실패 시 `config.json` 원본을 직접 읽어 리포트를 계속 생성합니다.
- 즉, `AutoConfig`가 실패해도 `ondevice/logs/inspect_report.json`에 실패 원인과 raw config 기반 sanity 결과가 남습니다.
