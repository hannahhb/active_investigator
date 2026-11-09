"""Start, check and download the Adaption AutoScientist run for the inflated-reporter organism.

This spends Adaption credits and uploads organism/train.csv (made-up data only) to Adaption.

    python3 organism/start_adaption.py models          # list base models (free)
    python3 organism/start_adaption.py upload          # upload train.csv only (then it shows in the web app)
    python3 organism/start_adaption.py start           # start training on the uploaded dataset (spends credits)
    python3 organism/start_adaption.py status          # check the run
    python3 organism/start_adaption.py download        # save the LoRA adapter to organism/adapter.tgz

The key is read from ai_village/.env and never printed. The run id is kept in organism/run.json.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ENV = pathlib.Path(os.environ.get("AI_VILLAGE_DIR", HERE.parent.parent / "ai_village")) / ".env"  # same folder as the data
STATE = HERE / "run.json"
MODEL = "Qwen/Qwen3.5-4B"  # small and quick; confirm the id with `models` first


def client():
    for line in ENV.read_text().splitlines():
        if line.startswith("ADAPTION_API_KEY="):
            os.environ["ADAPTION_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    from adaption import Adaption

    return Adaption()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    c = client()
    if cmd == "models":
        for m in c.autoscientist.list_models():
            print(m)
    elif cmd == "upload":  # upload only, no training: the dataset then shows up in the AutoScientist page
        ds = c.datasets.upload_file(str(HERE / "train.csv"), processing_mode="raw",
                                    column_mapping={"prompt": "prompt", "completion": "completion"})
        print("uploaded, dataset", ds.dataset_id, "- waiting for import")
        c.datasets.wait_for_completion(ds.dataset_id)
        STATE.write_text(json.dumps({"dataset_id": ds.dataset_id}))
        print("ready: refresh the AutoScientist page")
    elif cmd == "start":
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        ds_id = sys.argv[2] if len(sys.argv) > 2 else state.get("dataset_id")  # or pass the id shown in the web app
        if not ds_id:
            sys.exit("run `upload` first, or pass the dataset id: start <dataset_id>")
        run = c.autoscientist.create(dataset_id=ds_id, model=MODEL, training_method="instruction")
        STATE.write_text(json.dumps({"dataset_id": ds_id, "run_id": run.id, "model": MODEL}))
        print("training started, run", run.id)
    elif cmd == "status":
        run_id = json.loads(STATE.read_text())["run_id"]
        print(c.autoscientist.get(run_id))
    elif cmd == "download":
        run_id = json.loads(STATE.read_text())["run_id"]
        with c.autoscientist.with_streaming_response.download(run_id) as r:
            r.stream_to_file(str(HERE / "adapter.tgz"))
        print("saved", HERE / "adapter.tgz")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
