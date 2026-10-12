"""Inspect a selected saved model without training or inference."""
from __future__ import annotations
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",choices=("ukdale-10s","ukdale-1s","tdha-waveform","plaid-resistive"))
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--sha256",required=True,help="SHA-256 from the trusted release manifest")
    args=parser.parse_args()
    from wattsign.selected_models import load_model
    model=load_model(args.case,args.checkpoint,expected_sha256=args.sha256)
    result={"case":args.case,"class":type(model).__name__,"loaded":True,"training":False,"inference":False}
    if hasattr(model,"parameters"):
        result["parameters"]=sum(p.numel() for p in model.parameters())
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
