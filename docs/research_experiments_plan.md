# 研究実験運用フロー計画書

本書は、本リポジトリでARC-AGI-2タスクに対する共有サブスペース学習を検証する際の、データ前処理から実験実行、後解析・報告に至るまでの一連の手順を標準化することを目的とする。以下では、担当者が途中から参加しても同じ結果を再現できるよう、必須の準備・実施・記録タスクを順を追って整理する。

### スクリプトによる自動化

`python -m src.scripts.run_research_plan` を実行すると、本書に記載された主要フローを段階的に自動実行できる。既定の設定ファイルは `configs/research_plan.yaml` であり、`--list-stages` で利用可能なステージを確認できる。特定ステージのみを走らせたい場合は `--stage <name>` を複数回指定し、実行前に `--dry-run` でコマンド内容を確認すると安全である。学習や評価はGPUリソースを消費するため、必要なステージだけを選択して実行すること。

---

## 1. 前提整理
- **対象リポジトリ構成（主要ディレクトリ）**
  - `data/`: 生データ、分割ファイル、前処理済み成果物。
  - `configs/`: データ前処理・学習・評価の設定ファイル群。
  - `src/`: データローダー、モデル、学習ループ、解析ユーティリティ。
  - `tests/`: ユニット/統合テスト。
  - `reports/`: 実験結果をまとめた図表・ノートブック。
  - `docs/`: 計画・ログ・知見まとめ（本書を含む）。
- **共通環境**
  - Pythonバージョンおよび依存は `pyproject.toml` に従う。
  - 乱数シードは基本 `20250214` を使用し、追加設定は `configs/seed.yaml` で管理する。
  - 実験ログは `logs/` 配下に集約（存在しない場合は作成）。

---

## 2. データ前処理手順
1. **生データ取得**
   - ARC-AGI-2 の公式配布物から `arc-agi_{split}_challenges.json` と `arc-agi_{split}_solutions.json`（`split` は `training`、`evaluation`、`test`）を取得し、`data/raw/` に配置する。`test` スプリットには解答が含まれないため、評価指標算出には `training`/`evaluation` を用いる。
   - 取得元URL、ダウンロード日、整合性チェック（ハッシュ値など）を `logs/data_acquisition.log` に追記する。
2. **分割・サンプリング準備**
   - 検証タスクIDリストを生成し `data/splits/val_tasks.json` として保存。
   - メタ適応評価用のタスクサンプルを `data/splits/meta_eval_test.json` に作成。サンプリングは固定シードで一度だけ実施し、実行スクリプトを `configs/data_prep.yaml` で参照する。
3. **K-shot整形処理**
   - `src/data/` 以下の前処理スクリプトを用いて各タスクを辞書形式に整形し、`data/processed/{task_id}.json` に保存。
   - 例数がKを超える場合はサンプリングされたインデックスを `data/splits/kshot_indices/{task_id}.json` に記録して再利用する。
4. **検証・ログ出力**
   - 前処理後に基本統計（例数分布、グリッドサイズ）を `reports/data_summary.md` に追記。
   - 実行コマンド、Gitコミットハッシュ、所要時間を `logs/data_preparation.log` に記録する。
5. **自動テスト**
   - `pytest tests/data` を走らせ、データローダーの整合性を確認。失敗した場合はログを添付して修正する。

---

## 3. 実験設定準備
1. **設定ファイルの作成**
   - 学習設定（バッチサイズ、学習率、損失重みなど）は `configs/train/*.yaml` に保存し、バリアントごとにファイルを分ける。
   - 評価・解析用設定は `configs/eval/*.yaml`、`configs/analysis/*.yaml` にまとめる。
2. **追跡用メタデータ**
   - 各実験は一意なID（例：`EXP-YYYYMMDD-xx`）を付与し、`docs/experiment_log.md` のテーブルに追記する。
   - 使用するコードブランチ、コミットハッシュ、設定ファイル、GPU情報を合わせて記録する。
3. **事前確認**
   - `pytest` と `ruff` などの静的解析を実行し、基本的な実装不具合を除去する。
   - 重いテストが困難な場合でも `pytest tests/unit` の最低限は通すこと。

---

## 4. 学習・実験実行
1. **学習ジョブ起動**
   - 推奨コマンド雛形：
     ```bash
     python -m src.train.run \
       --config configs/train/baseline.yaml \
       --data-root data/processed \
       --output-dir reports/experiments/EXP-YYYYMMDD-xx
     ```
   - 進捗はTensorBoardまたは独自ロギング (`logs/`) で追跡。GPU利用率などは `nvidia-smi` ログを適宜保存する。
2. **チェックポイント管理**
   - チェックポイントは `reports/checkpoints/EXP-YYYYMMDD-xx/` に保存し、最新のものを `latest.ckpt` としてシンボリックリンク化する。
   - ストレージ節約のため古いチェックポイントを整理する際は `docs/experiment_log.md` に削除理由を記載する。
3. **モニタリング**
   - 重要指標（タスク正解率、ピクセル精度、損失推移）は `reports/metrics/EXP-YYYYMMDD-xx.csv` に書き出す。
   - 異常（損失発散、NaN発生等）を検知した場合は即時停止し、`logs/incident.log` に詳細を残す。

---

## 5. 評価および後解析
1. **基本評価**
   - 検証・テストデータに対し `python -m src.eval.run --config configs/eval/default.yaml --checkpoint <path>` を実行。
   - 結果サマリを `reports/summaries/EXP-YYYYMMDD-xx.json` に保存する。
2. **メタ適応評価**
   - 指定サブセットに対するfew-shot微調整ジョブを `python -m src.eval.meta_adapt` で実行し、性能推移を `reports/meta/` に保存。
3. **表現解析**
   - `src.analysis` のスクリプト群を用いてSAE特徴やCKA計算を行い、図表を `reports/figures/EXP-YYYYMMDD-xx/` に格納。
   - 解析手順と主要観察結果は `docs/findings.md` に追記し、追跡性を確保する。
4. **再現性確保**
   - 使用した乱数シード、設定ファイル、チェックポイントを明記した`README`的メモを `reports/experiments/EXP-YYYYMMDD-xx/README.md` に残す。

---

## 6. レポーティングとナレッジ共有
1. **ドキュメント更新**
   - 実験の開始・完了時に `docs/experiment_log.md` と `docs/ablation_summary.md` を更新する。
   - 重要な洞察は `docs/findings.md` に段落を追加し、関連する図表へのパスを明記。
2. **報告テンプレート**
   - 週次進捗共有や論文ドラフトへ活用するため、`reports/weekly/` に `EXP-YYYYMMDD-xx_report.md` を作成する。
   - レポートには背景、設定、主結果、課題、次アクションを含める。
3. **バージョン管理**
   - 主要な設定・コード変更はPull Request単位で管理し、PR本文に実験IDと主要指標の差分を記載する。
   - 本ドキュメントの内容が変更された場合、変更理由と対象実験を冒頭に追記する。

---

## 7. トラブルシューティングとバックアップ
- **データ破損時**：`data/raw/` のハッシュを `logs/data_acquisition.log` と照合し、差異があれば再ダウンロード。
- **学習失敗時**：設定（学習率、損失重み）を `configs/train/` 内で調整し、再実行時は新しい実験IDを付与。
- **解析スクリプト不具合**：`tests/analysis` にユニットテストを追加し再発防止。既存結果ファイルは`reports/archive/`に退避し、バージョンを明示。
- **長期保存**：重要なチェックポイントとレポートは外部ストレージへ同期し、同期ログを `logs/backup.log` に記録する。

---

## 8. 変更履歴
- 2025-02-14: 初版作成（データ前処理〜後解析の標準フローを定義）。
