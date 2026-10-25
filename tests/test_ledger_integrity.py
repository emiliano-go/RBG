from rbg.db import Store


def test_ledger_chain_verifies_and_detects_tampering(tmp_path):
    store = Store(str(tmp_path))
    store.append_ledger({"task_id": "t1", "component": "c", "result": {"x": 1}})
    store.append_ledger({"task_id": "t2", "component": "c", "result": {"x": 2}})
    assert store.verify_ledger()["ok"] is True
    store.conn.execute("UPDATE ledger SET result='{\"x\": 99}' WHERE task_id='t1'")
    verdict = store.verify_ledger()
    assert verdict["ok"] is False and verdict["first_bad_id"] == 1
    store.close()
