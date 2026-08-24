#!/usr/bin/env bash
# Progress of the Qwen3.8-27B resume: how many of the 34 missing runs are done.
cd "$(dirname "${BASH_SOURCE[0]}")"
python3.11 - <<'PY'
from pathlib import Path
EXPECT={('mceval','java'):53,('mceval','python'):50,
        ('codereval','java'):230,('codereval','python'):230}
done=todo=0; pending=[]
for bench in ('mceval','codereval'):
    for reg in ('pass@1_t1','pass@5_t1'):
        root=Path(f'experiments_results_{bench}/{reg}/predictions/Qwen3.8_27b')
        for lang in ('java','python'):
            for p in range(5):
                f=root/f'qwen3.8-27b-original_{lang}_retrieval_3-shot_{p}'/'predictions.jsonl'
                if f.is_file() and sum(1 for _ in f.open())>=EXPECT[(bench,lang)]:
                    done+=1
                else:
                    todo+=1; pending.append(f"{bench[:2]}/{reg[-4:]}/{lang[:2]}/p{p}")
total=done+todo
bar=int(40*done/total)
print(f"[{'#'*bar}{'.'*(40-bar)}] {done}/{total} runs  ({100*done/total:.0f}%)")
if pending: print("pending:", " ".join(pending[:12]) + (" ..." if len(pending)>12 else ""))
PY
