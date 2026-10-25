"""List Amazon Nova / Anthropic models and inference profiles available to this account in a region,
and try one Converse tool-use call against the configured model.

  python scripts/check_bedrock.py --region eu-north-1 [--model eu.amazon.nova-lite-v1:0]
"""
import argparse
import os
import sys

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-north-1"))
    ap.add_argument("--model", default=os.environ.get("BEDROCK_MODEL_ID", "eu.amazon.nova-lite-v1:0"))
    a = ap.parse_args()
    br = boto3.client("bedrock", region_name=a.region)
    models = [m for prov in ("Amazon", "Anthropic")
              for m in br.list_foundation_models(byProvider=prov)["modelSummaries"]
              if "nova" in m["modelId"] or "anthropic" in m["modelId"]]
    print(f"Nova / Anthropic foundation models in {a.region}:")
    for m in sorted(models, key=lambda m: m["modelId"]):
        print(f"  {m['modelId']:<55} {','.join(m.get('inferenceTypesSupported', []))}")
    try:
        profiles = br.list_inference_profiles()["inferenceProfileSummaries"]
        print("Inference profiles (use these IDs when the model needs cross-region inference):")
        for p in profiles:
            if "nova" in p["inferenceProfileId"] or "anthropic" in p["inferenceProfileId"]:
                print(f"  {p['inferenceProfileId']}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not list inference profiles: {e})")
    if a.model:
        os.environ["BEDROCK_MODEL_ID"] = a.model
        os.environ["BEDROCK_REGION"] = a.region
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from unmask.agent.bedrock import PolicyError, bedrock_policy
        from unmask.agent.policy import build_state
        from unmask.scoring import load_config, session_risk
        cfg = load_config()
        clips = [{"challenge": "passive", "usable": True, "features": {"seam_flicker": 1.6, "seam_edge": 1.35}}]
        state = build_state(clips, session_risk(clips, cfg), cfg, ["head_turn", "hand_pass", "light_change"])
        try:
            d = bedrock_policy(state, cfg, False)
            print(f"\nConverse tool call OK with {d['model_id']} (configured {a.model}): {d['action']} {d.get('challenge', '')} ({d['model_latency_ms']:.0f} ms)")
            print(f"  reason: {d['reason']}")
        except PolicyError as e:
            print(f"\nConverse call FAILED with {a.model}: {e}")
            print("Enable model access in the Bedrock console, or pick an ID/profile from the list above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
