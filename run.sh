xformer="linformer"

PYTHONPATH="$(pwd)":"$PYTHON_PATH" python lra_benchmarks/listops/train.py \
      --config="lra_benchmarks/listops/configs/${xformer}_base.py" \
      --model_dir="/data/lra_results/${xformer}" \
      --task_name=basic \
      --data_dir=/data/lra_data/listops-1000/
