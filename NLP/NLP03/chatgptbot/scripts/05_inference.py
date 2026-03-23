import torch, pickle, sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
spec = importlib.util.spec_from_file_location("train", os.path.join(os.path.dirname(__file__), "04_train.py"))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(config.TOKENIZER_PATH, "rb") as f:
    tok = pickle.load(f)
word2idx = tok["word2idx"] if isinstance(tok, dict) else tok.word2idx
idx2word = tok["idx2word"] if isinstance(tok, dict) else tok.idx2word
vocab_size = len(word2idx)
model = tm.GPTModel(vocab_size=vocab_size, d_model=config.TRANSFORMER_D_MODEL, nhead=config.TRANSFORMER_NHEAD, num_layers=config.TRANSFORMER_NUM_LAYERS, dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD, dropout=0.0, max_seq_length=config.MAX_SEQ_LENGTH)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()
print(model)

from konlpy.tag import Okt
_okt = Okt()
def okt_tokenize(text):
    return _okt.morphs(text)

def generate(prompt, max_new_tokens=30):
    ids = [word2idx.get(t, 1) for t in okt_tokenize(prompt)]
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(device)
            logits = model(x, None, mask)
            nt = logits[0, -1, :].argmax().item()
            if idx2word.get(nt) in ["<end>", "<pad>"]: break
            x = torch.cat([x, torch.tensor([[nt]]).to(device)], dim=1)
    return " ".join([idx2word.get(i, "<unk>") for i in x[0].tolist()])

print("\n=== GPT 챗봇 대화 모드 (종료: q) ===")
while True:
    p = input("\n입력: ").strip()
    if p.lower() == 'q':
        break
    if not p:
        continue
    print(f"출력: {generate(p)}")