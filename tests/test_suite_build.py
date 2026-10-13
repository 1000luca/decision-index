import json

from decision_index.suite.build.adapters_mechanical import build_sgd
from decision_index.suite.build.layout import Layout


def test_sgd_history_stops_at_its_own_turn(tmp_path):
    layout = Layout(tmp_path)
    base = layout.repos / "sgd/test"
    base.mkdir(parents=True)
    schema = [{"service_name": "Svc", "intents": [{"name": "Book"}]}]
    (base / "schema.json").write_text(json.dumps(schema))
    turns = []
    for i in range(3):
        turns.append({"speaker": "USER", "utterance": f"u{i}", "frames": [{"service": "Svc", "state": {"active_intent": "Book"}}]})
        turns.append({"speaker": "SYSTEM", "utterance": f"s{i}", "frames": []})
    (base / "dialogues_001.json").write_text(json.dumps([{"dialogue_id": "d1", "turns": turns}]))
    n, _ = build_sgd(layout)
    rows = [json.loads(line) for line in (layout.normalized / "SGD-service-given-intent.jsonl").read_text().splitlines()]
    assert n == len(rows) == 3
    for row in rows:
        ti = row["metadata"]["provenance"]["turn_index"]
        history = row["state"]["history"]
        assert len(history) == ti + 1
        assert history[-1] == {"speaker": "USER", "utterance": turns[ti]["utterance"]}
