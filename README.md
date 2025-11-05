# ARC Mutual Concept Loss リポジトリ

## プロジェクト概要
ARC Mutual Concept Loss は、Abstraction and Reasoning Corpus (ARC) の拡張データセットである **ARC-AGI-2** を対象に、
タスク横断で共有される潜在概念を抽出・再利用させる学習手法を検証する研究プロジェクトです。サブスペース共有正則化と疎オートエンコーダ
(SAE) を組み合わせた "mutual concept loss" を用い、

- 未見タスクに対する In-Context Learning 精度の向上
- few-shot でのメタ適応効率の改善
- 学習挙動の解析 (CKA, Grassmann 距離, SAE 可視化)

を同時に達成できるかを検証します。本リポジトリにはデータ前処理パイプライン、トレーニングループ、評価・解析スクリプト、および実験ログ管理のための補助ツールが含まれています。

## ディレクトリ構成
- `configs/` : トレーニング・評価・メタ適応・アブレーション実験の YAML 設定ファイル。
- `data/` : `raw/` に ARC-AGI-2 の JSON を配置し、`processed/`・`splits/` に前処理結果が生成されます。
- `docs/` : 実験計画、所見、アブレーション結果のまとめなど研究ドキュメント。
- `reports/` : トレーニング履歴や評価指標を JSON/画像で保存する出力先。
- `src/` : 主要な Python 実装。`data/` (前処理)、`train/` (データローダ・損失・トレーナ)、`models/` (ARC in-context モデル)、`losses/` (mutual concept loss 構成要素)、`analysis/` (表現解析)、`scripts/` (CLI エントリポイント) などで構成されています。
- `tests/` : データパイプラインやトレーニング補助機能のユニットテスト。

## 必要要件
- Python 3.10
- CUDA 対応 GPU (推奨。CPU のみでも実行可能ですが学習時間が長くなります)
- `pip`, `venv` など標準的な Python ツールチェーン

## セットアップ手順
1. リポジトリをクローンします。
   ```bash
   git clone <this-repository-url>
   cd arc-mutual-concept-loss
   ```
2. 仮想環境を作成して有効化します。
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows の場合は .venv\Scripts\activate
   ```
3. 依存関係をインストールします。
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
   開発用ツール (pytest, mypy, ruff など) も利用する場合は以下を実行してください。
   ```bash
   pip install -e .[test]
   ```

## データ準備
1. ARC-AGI-2 の JSON データを `data/raw/` に配置します。リポジトリには `ARC-AGI-2-main/` ディレクトリが含まれているため、必要に応じてそこから `arc-agi_*` ファイルをコピーできます。
2. 前処理スクリプトを実行して k-shot 対応のデータセットと分割情報を生成します。
   ```bash
   python -m src.data.preprocess --config configs/data_prep.yaml
   ```
   - `data/processed/` にタスク単位の JSON (k-shot 例・テスト例・メタデータ) が生成されます。
   - `data/splits/` に学習/検証/メタ評価タスクのリストが保存されます。
   - 実行ログは `logs/data_preparation.log` に追記されます。
3. `configs/data_prep.yaml` を編集することで、k-shot 数や検証データ比率、ランダムシードなどを調整できます。

## 学習・評価ワークフロー
### トレーニング (アブレーション実験スイート)
アブレーション設定をまとめて実行する例:
```bash
python -m src.scripts.run_ablations --configs-dir configs/ablations --summary-path reports/ablations/summary.json
```
- `configs/train.yaml` を基準に、各アブレーション設定がマージされます。
- チェックポイントは `checkpoints/`、TensorBoard ログは `logs/train/tensorboard/`、学習履歴は `logs/train/history.jsonl` に保存されます。

### 評価 (In-Context 推論性能)
学習済みモデルのチェックポイントを用いて評価を実行します。
```bash
python -m src.scripts.run_eval --config configs/eval.yaml
```
- 指定した分割 (デフォルトでは検証またはメタ評価) に対する正解率などの指標を標準出力と `reports/eval/` 以下の JSON に保存します。

### 評価結果の可視化と予測エクスポート
推論結果を JSON と画像で保存したい場合は、以下の可視化付きスクリプトを利用してください。
```bash
python -m src.scripts.run_eval_with_visualization \
    --config configs/eval.yaml \
    --predictions-dir reports/ic_eval/predictions \
    --figures-dir reports/ic_eval/figures
```
- `--predictions-dir` に指定したディレクトリへタスク別の予測 JSON (`predictions_*.json`) が保存されます。
- `--figures-dir` にはタスクごとのサブディレクトリが作成され、クエリ入力・モデル予測・正解の 3 枚を並べた PNG が出力されます。
- `visualization_index.json` に、保存された指標値と画像パスの一覧が記録されます。`--max-visualized-tasks` や `--max-queries-per-task` で可視化数を絞ることも可能です。

### テストチャレンジへの推論と提出ファイル生成
ARC-AGI テストチャレンジ (`data/raw/arc-agi_test_challenges.json`) を入力に、提出用の `submission.json` を作成する場合は次のスクリプトを利用します。
```bash
python -m src.scripts.generate_submission \
    --checkpoint checkpoints/latest.pt \
    --config configs/eval.yaml \
    --challenge-path data/raw/arc-agi_test_challenges.json \
    --output submission.json
```
- `--config` にはモデル構成 (主に `evaluation.model` セクション) が記載された YAML を指定します。追加のモデルハイパーパラメータを JSON 文字列で渡したい場合は `--model-kwargs` を利用できます。
- 推論は 1 タスクずつ順に実行され、各テスト例に対して `attempt_1` と `attempt_2` の 2 つの出力グリッドが生成されます (後者は学習例の出力サイズを参考にした推測です)。
- 出力先ディレクトリが存在しない場合は自動的に作成されます。

### Few-Shot メタ適応
LoRA/Adapter を用いたメタ適応を評価する場合:
```bash
python -m src.scripts.run_meta_adaptation --config configs/meta_adaptation.yaml
```
- 1 タスクずつ適応し、適応前後の指標を `reports/meta_adaptation/` に記録します。

### 研究計画の一括実行
ドキュメント更新や品質チェックを含む全体フローを自動化する場合はリサーチプラン実行スクリプトを利用できます。
```bash
python -m src.scripts.run_research_plan --config configs/research_plan.yaml
```

## テストとコード品質チェック
開発時には以下のコマンドで品質を確認できます。
```bash
pytest              # ユニットテスト
ruff check src tests  # コードスタイル / Lint
mypy src            # 型チェック
```
`pip install -e .[test]` を実行していれば、必要な追加依存関係がインストールされています。

## 参考ドキュメント
- 実験計画: `docs/experiment_plan.md`
- 実装計画と進捗ノート: `docs/implementation_plan.md`, `docs/experiment_log.md`
- 解析結果サマリ: `docs/findings.md`, `reports/analysis/`

研究の進行に合わせて README や `docs/` 内のファイルを随時更新してください。
