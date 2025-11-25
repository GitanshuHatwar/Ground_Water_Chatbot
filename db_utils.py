def flatten_json_to_text(json_data: dict) -> str:
    lines = []
    for k, v in json_data.items():
        lines.append(f"{k.replace('_',' ')}: {v}")
    return "\n".join(lines)
