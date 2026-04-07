import argparse
import base64
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


ALLOWED = {"N", "C", "S", "Unknown"}


def normalize_label(v: str) -> str:
    s = str(v).strip().lower()
    mapping = {"n": "N", "c": "C", "s": "S", "unknown": "Unknown"}
    return mapping.get(s, "Unknown")


def safe_note(existing_note: str, add_text: str) -> str:
    base = (existing_note or "").strip()
    if not base:
        return add_text
    if add_text in base:
        return base
    return f"{base}; {add_text}"


def image_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def prompt_text() -> str:
    return (
        "Classify this spine X-ray into one label: N, C, S, Unknown.\n"
        "Definitions:\n"
        "- N: non-scoliosis / normal (no meaningful lateral curve)\n"
        "- C: one dominant scoliosis curve\n"
        "- S: two dominant opposite scoliosis curves\n"
        "- Unknown: unclear image or ambiguous anatomy\n\n"
        "Return STRICT JSON object only:\n"
        '{"label":"N|C|S|Unknown","confidence":0.0-1.0,"reason":"short"}'
    )


def build_request_line(model: str, custom_id: str, image_data_url: str):
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text()},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def cmd_submit(args):
    csv_path = Path(args.csv).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    df = pd.read_csv(csv_path)
    for c in ["image_path", "image_name", "label", "confidence", "notes"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["label"] = df["label"].fillna("").astype(str)
    work_df = df[df["label"].str.strip() == ""].copy()
    if args.max_rows > 0:
        work_df = work_df.iloc[: args.max_rows]
    if work_df.empty:
        print("No empty labels found. Nothing to submit.")
        return

    rows = work_df.to_dict(orient="records")
    req_path = out_dir / "openai_batch_requests.jsonl"
    map_path = out_dir / "openai_batch_map.json"

    mapping = {}
    with req_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            image_rel = str(row["image_path"])
            image_name = str(row["image_name"])
            image_abs = (dataset_root / image_rel).resolve()
            if not image_abs.exists():
                continue

            custom_id = f"row_{i}_{image_name}"
            line = build_request_line(
                model=args.model,
                custom_id=custom_id,
                image_data_url=image_to_data_url(image_abs),
            )
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            mapping[custom_id] = {"image_name": image_name}

    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Request JSONL: {req_path}")
    print(f"Mapping JSON: {map_path}")

    client = OpenAI()
    with req_path.open("rb") as f:
        in_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "xray-prelabel-ncsu"},
    )

    info = {
        "batch_id": batch.id,
        "input_file_id": in_file.id,
        "request_jsonl": str(req_path),
        "mapping_json": str(map_path),
        "csv_path": str(csv_path),
    }
    info_path = out_dir / "openai_batch_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Batch submitted. batch_id={batch.id}")
    print(f"Info file: {info_path}")
    print("Run the apply command after batch status is 'completed'.")


def cmd_apply(args):
    info_path = Path(args.info_json).resolve()
    if not info_path.exists():
        raise FileNotFoundError(f"Info JSON not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    csv_path = Path(info["csv_path"]).resolve()
    map_path = Path(info["mapping_json"]).resolve()
    out_dir = info_path.parent
    output_jsonl = out_dir / "openai_batch_output.jsonl"

    client = OpenAI()
    batch = client.batches.retrieve(info["batch_id"])
    print(f"Batch status: {batch.status}")
    if batch.status != "completed":
        print("Batch is not completed yet. Try again later.")
        return
    if not batch.output_file_id:
        raise RuntimeError("Batch completed but no output_file_id found.")

    content = client.files.content(batch.output_file_id).text
    output_jsonl.write_text(content, encoding="utf-8")
    print(f"Output saved: {output_jsonl}")

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].fillna("").astype(str)
    df["confidence"] = df["confidence"].fillna("").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)

    parsed = 0
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")
        if custom_id not in mapping:
            continue

        # Batch output shape for chat/completions
        body = obj.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            continue
        text = choices[0].get("message", {}).get("content", "")
        try:
            payload = json.loads(text)
        except Exception:
            continue

        image_name = mapping[custom_id]["image_name"]
        label = normalize_label(payload.get("label", "Unknown"))
        if label not in ALLOWED:
            label = "Unknown"
        conf = float(payload.get("confidence", 0.0))
        reason = str(payload.get("reason", "")).strip()

        idx = df.index[df["image_name"] == image_name].tolist()
        if not idx:
            continue
        ridx = idx[0]
        # Fill only empty labels.
        if str(df.at[ridx, "label"]).strip() != "":
            continue

        df.at[ridx, "label"] = label
        df.at[ridx, "confidence"] = f"{conf:.3f}"
        if conf < args.low_conf_threshold:
            reason = f"low_conf_from_ai; {reason}".strip("; ").strip()
        if reason:
            df.at[ridx, "notes"] = safe_note(df.at[ridx, "notes"], reason)
        parsed += 1

    backup = csv_path.with_suffix(".before_openai_batch_backup.csv")
    if not backup.exists():
        pd.read_csv(csv_path).to_csv(backup, index=False)
        print(f"Backup created: {backup}")

    df.to_csv(csv_path, index=False)
    print(f"Updated CSV: {csv_path}")
    print(f"Rows updated from batch output: {parsed}")


def main():
    load_env_file(Path(".env").resolve())
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to .env or set it in your shell."
        )

    parser = argparse.ArgumentParser(
        description="OpenAI Batch API pre-label flow for N/C/S/Unknown."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="Create and submit batch job.")
    p_submit.add_argument("--csv", type=str, default="labels/pilot_labels_template.csv")
    p_submit.add_argument("--dataset-root", type=str, default="spinal-ai2024")
    p_submit.add_argument("--model", type=str, default="gpt-4o-mini")
    p_submit.add_argument("--max-rows", type=int, default=0)
    p_submit.add_argument("--out-dir", type=str, default="batch_jobs/openai_prelabel")
    p_submit.set_defaults(func=cmd_submit)

    p_apply = sub.add_parser("apply", help="Download completed batch and apply labels.")
    p_apply.add_argument(
        "--info-json",
        type=str,
        default="batch_jobs/openai_prelabel/openai_batch_info.json",
    )
    p_apply.add_argument("--low-conf-threshold", type=float, default=0.60)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
