import os
import json

runs = [
    ("/home/james/ThinkGeo/opencompass/outputs/default/Qwen1.5WithWeights/predictions/qwen1.5-7b-chat", "Qwen1.5WithWeights"),
    ("/home/james/ThinkGeo/opencompass/outputs/default/Qwen2.5WithWeights/predictions/qwen2.5-7b-instruct", "Qwen2.5WithWeights"),
    ("/home/james/ThinkGeo/opencompass/outputs/default/MistralWithWeIGHTS/predictions/mistral-7b-instruct", "MistralWithWeIGHTS"),
]

for pred_dir, run_name in runs:
    files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    total_tool_call_errors = 0
    total_classifier_rejections = 0
    correct_rejections = 0

    for fname in files:
        with open(os.path.join(pred_dir, fname), "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        for k, v in data.items():
            pred_steps = v.get("prediction", [])
            gold_steps = v.get("gold", [])
    def find_prediction_files(base_dirs):
        files = []
        for base in base_dirs:
            for root, _, fnames in os.walk(base):
                for fname in fnames:
                    if fname.endswith('.json'):
                        files.append(os.path.join(root, fname))
        return files

    def enumerate_errors(files):
        error_counts = {}
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
            except Exception:
                continue
            for k, v in data.items():
                pred = v.get('prediction', [])
                for step in pred:
                    err = step.get('error')
                    if err:
                        etype = err.get('type', 'UNKNOWN')
                        error_counts[etype] = error_counts.get(etype, 0) + 1
        return error_counts

    if __name__ == '__main__':
        base_dirs = [
            '/home/james/ThinkGeo/opencompass/outputs/default/Qwen1.5WithWeights/predictions',
            '/home/james/ThinkGeo/opencompass/outputs/default/Qwen2.5WithWeights/predictions',
            '/home/james/ThinkGeo/opencompass/outputs/default/MistralWithWeIGHTS/predictions',
        ]
        files = find_prediction_files(base_dirs)
        error_counts = enumerate_errors(files)
        print('Tool call error types and counts:')
        for etype, count in error_counts.items():
            print(f'{etype}: {count}')
            for idx, step in enumerate(pred_steps):
                # Check for tool call error
                if "error" in step:
                    total_tool_call_errors += 1
                    if step["error"].get("type") == "CLASSIFIER_REJECTION":
                        total_classifier_rejections += 1
                        # Check if rejected tool is not the ground truth tool
                        rejected_tool = step["tool_calls"][0]["function"]["name"] if "tool_calls" in step and step["tool_calls"] else None
                        gt_tool = None
                        if idx < len(gold_steps):
                            gt_tool_calls = gold_steps[idx].get("tool_calls", [])
                            if gt_tool_calls:
                                gt_tool = gt_tool_calls[0]["function"]["name"]
                        if rejected_tool and gt_tool and rejected_tool != gt_tool:
                            correct_rejections += 1

    print(f"\nRun: {run_name}")
    print(f"Total tool call errors: {total_tool_call_errors}")
    print(f"Total classifier rejections: {total_classifier_rejections}")
    print(f"Correct classifier rejections: {correct_rejections}")
    if total_classifier_rejections > 0:
        print(f"Percentage correct: {correct_rejections / total_classifier_rejections * 100:.2f}%")
    else:
        print("No classifier rejections found.")