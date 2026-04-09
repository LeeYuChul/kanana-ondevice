"""Kanana tokenizer.json (BPE) → SentencePiece .model (protobuf) 변환"""
import json
import os
from sentencepiece import sentencepiece_model_pb2

with open('/tmp/kanana_tokenizer/tokenizer.json') as f:
    tok_data = json.load(f)

vocab = tok_data['model']['vocab']  # dict: token -> id
added_tokens = {item['content']: item['id'] for item in tok_data.get('added_tokens', [])}

# Reverse vocab: id -> token
id_to_token = {v: k for k, v in vocab.items()}
for content, tid in added_tokens.items():
    id_to_token[tid] = content

print(f"Vocab size: {len(id_to_token)}")
print(f"Max id: {max(id_to_token.keys())}")

# SentencePiece protobuf 생성
m = sentencepiece_model_pb2.ModelProto()
m.trainer_spec.model_type = 1  # BPE
m.trainer_spec.vocab_size = 128256
m.trainer_spec.bos_id = 128000
m.trainer_spec.eos_id = 128001
m.trainer_spec.unk_id = 0  # unk 필수 - 첫 번째 토큰으로 설정
m.trainer_spec.pad_id = -1
m.normalizer_spec.name = "identity"
m.normalizer_spec.add_dummy_prefix = False

# vocab 추가
SentencePiece = sentencepiece_model_pb2.ModelProto.SentencePiece
max_id = max(id_to_token.keys())
for i in range(max_id + 1):
    piece = m.pieces.add()
    token = id_to_token.get(i, f"<unk_{i}>")
    piece.piece = token
    piece.score = 0.0
    if i == 0:  # unk
        piece.piece = "<unk>"
        piece.type = SentencePiece.UNKNOWN
    elif i == 128000:  # BOS
        piece.type = SentencePiece.CONTROL
    elif i == 128001:  # EOS
        piece.type = SentencePiece.CONTROL
    elif i not in id_to_token:
        piece.type = SentencePiece.CONTROL
    else:
        piece.type = SentencePiece.NORMAL

output_path = '/tmp/kanana_tokenizer/kanana_tokenizer.model'
with open(output_path, 'wb') as f:
    f.write(m.SerializeToString())

size_kb = os.path.getsize(output_path) / 1024
print(f"Saved: {output_path} ({size_kb:.0f} KB)")
