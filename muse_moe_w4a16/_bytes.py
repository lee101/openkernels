def nbytes(data):
    cfg = data["config"]; ids = data["topk_ids"]; lo = cfg["local_start"]
    e = ids[(ids >= lo) & (ids < lo + cfg["local_count"])].unique().numel()
    return e * 3 * 4096 * 4096 * (0.5 + 1 / 16) + data["hidden_states"].numel() * 4
