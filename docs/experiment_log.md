# 評価ログ

本ファイルでは In-Context 評価の実行条件および得られた指標の記録方法をまとめる。

## 評価ワークフロー
- `configs/eval.yaml` に評価用のデータモジュール設定とチェックポイントパスを記述する。
- `python -m src.scripts.run_eval --config configs/eval.yaml` を実行すると、指定した分割のタスクに対して推論を行い、
  - `reports/ic_eval/` 配下に JSON 形式で指標を保存する。
  - 必要に応じて `predictions_dir` で指定したディレクトリに予測グリッドを保存する。
- 保存される主な指標は以下の通り。
  - `task_top1` / `task_top3`: タスク単位の Top-k 完全一致率。
  - `pixel_accuracy`: 画素単位の精度。
  - `mean_iou`: クラスごとの IoU を平均した値。
  - `exact_match_rate`: クエリごとの完全一致率。

## ログ更新手順
1. `run_eval.py` を実行し、`reports/ic_eval/*.json` が生成されたことを確認する。
2. 実験条件（チェックポイント、データ分割、上位指標など）を本ファイルに追記する。
3. 重要な結果については、追加で `docs/experiment_plan.md` に反映する。
