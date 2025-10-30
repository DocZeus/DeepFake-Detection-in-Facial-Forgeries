# deepfake_hodff/compress.py
import torch
import torch.nn.utils.prune as prune
from deepfake_hodff.models import HODFF_DD
from deepfake_hodff.config import *

def prune_linear(module, amount=0.3):
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')

def quantize_dynamic(model):
    q = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8
    )
    return q

def main():
    model = HODFF_DD()
    model.load_state_dict(torch.load(CKPT_DIR/"hodff_dd.pt", map_location="cpu"))
    prune_linear(model.temporal, amount=0.2)
    qmodel = quantize_dynamic(model)
    torch.save(qmodel.state_dict(), CKPT_DIR/"hodff_dd_quantized.pt")
    print("Saved quantized model.")

if __name__ == "__main__":
    main()
