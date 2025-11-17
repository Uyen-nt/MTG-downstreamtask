# gram/scripts/fine_tune_mimic.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from gram.model.gram import GRAM, load_tree, pad_batch

SYNTH_MODEL = "gram/data/synth_train_best.pt"
TREE_PREFIX = "gram/data/mimic3_tree"
MIMIC_FILE = "gram/data/mimic.seqs"

FT_OUT = "gram/data/finetuned.pt"

def clean_seqs(seqs):
    new = []
    for p in seqs:
        v = [x for x in p if len(x) > 0]
        if len(v)>=2:
            new.append(v)
    return new


def train_finetune():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== FINE-TUNE GRAM WITH MIMIC3 REAL DATA =====")

    seqs = pickle.load(open(MIMIC_FILE,"rb"))
    seqs = clean_seqs(seqs)

    num_codes = max(max(max(v) for v in p) for p in seqs)+1
    num_classes = num_codes

    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    all_idx=[]
    for L,A in zip(tree_leaves, tree_anc): 
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types","rb"))
    all_idx.append(max(types.values()))

    max_index = max(all_idx)

    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_anc,
        max_index_in_tree=max_index,
        device=device
    ).to(device)

    model.load_state_dict(torch.load(SYNTH_MODEL))
    print("Loaded pretrained (synthetic) model.")

    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    X = [p[:-1] for p in seqs]
    Y = [p[1:] for p in seqs]

    for epoch in range(10):
        model.train()
        tot = 0

        for i in range(0,len(X),32):
            xb = X[i:i+32]
            yb = Y[i:i+32]

            xpad,_,mask,len_ = pad_batch(xb, num_classes, num_codes, device)
            _,ypad,_,_ = pad_batch(yb, num_classes, num_codes, device)

            pred = model(xpad,mask)

            ylab = ypad.argmax(dim=-1)   # (T,B)
            loss_step = loss_fn(pred.permute(0,2,1), ylab)
            loss = (loss_step*mask).sum() / mask.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

            tot += loss.item()

        print(f"[Epoch {epoch+1}] Loss={tot:.4f}")

    torch.save(model.state_dict(), FT_OUT)
    print("Saved fine-tuned model →", FT_OUT)


if __name__=="__main__":
    train_finetune()
