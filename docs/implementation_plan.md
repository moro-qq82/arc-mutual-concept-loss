# 実装計画：共有サブスペースXによるARC-AGI-2一般化検証

## 1. ゴールと成果物
- `ARC-AGI-2` のタスクを対象に、共有サブスペース正則化と疎オートエンコーダ（SAE）を組み込んだ K-shot In-Context Learning モデルを実装する。
- データ前処理、学習・評価スクリプト、メタ適応ループ、表現解析ツールを整備し、再現性のある実験環境を提供する。
- 実験ログ、チェックポイント、解析結果を再利用可能な形で整理する。

## 2. 前提・依存関係
- **データ**：`ARC-AGI-2` 配布物の JSON ディレクトリ（`training/*.json`, `evaluation/*.json` など）または JSONL ファイルを `data/raw/` 相当のパスに配置済みであること。
- **ハードウェア**：単一 GPU（16GB 以上推奨、AMP/BF16 が利用可能）と十分なストレージ。
- **ソフトウェア**：PyTorch 2.2 以降、`torchvision`、`numpy`、`scikit-learn`、`hydra`、`tensorboard` 等の依存パッケージ。
- **設定管理**：`configs/` ディレクトリに Hydra 形式の設定群を用意し、`logs/` に前処理・学習ログ、`checkpoints/` にモデルを保存する。

## 3. 実装タスク詳細
### 3.1 データ準備
1. `src/data/raw_loader.py`
   - ARC-AGI-2 の JSON ディレクトリ（`training/*.json` など）および JSONL 配布形式の両方からタスクを読み出し、入出力グリッドとメタデータを Python オブジェクトへ変換するローダーを実装。
   - 乱数シード `20250214` を用いた再現性確保。
2. `src/data/preprocess.py`
   - K-shot 整形、`data/processed/{task_id}.json` 書き出し、`data/splits/` 配下ファイル生成（`val_tasks.json`, `meta_eval_test.json`, `kshot_indices/`）。
   - タスクごとのシードを BLAKE2b で派生させ、K-shot の抽出順を安定化。
   - 実行ログを `logs/data_preparation.log` に追記し、使用設定を `configs/data_prep.yaml` に保持。
3. ユニットテスト：`tests/test_data_pipeline.py` を作成し、サンプルタスクのロード・整形結果を検証。

### 3.2 モデルアーキテクチャ
1. `src/models/grid_encoder.py`
   - 入力グリッドを埋め込みベクトルへ変換する CNN または ResNet 風のエンコーダを実装。
2. `src/models/context_encoder.py`
   - K 個の例ペアを受け取りタスク表現 `h_task` を生成する Transformer ベースのエンコーダ。
   - マスク処理とポジション埋め込みを含める。
3. `src/models/solver.py`
   - `h_task` とテスト入力を条件にグリッドを生成するデコーダ（Transformer Decoder もしくは U-Net）。
4. `src/models/sae.py`
   - `h_task` を疎表現 `z` に射影し再構成する SAE。L1 正則化と再構成 MSE を同時に扱う。
5. モジュール統合：`src/models/ic_model.py` で上記各構成要素を統合した `ARCInContextModel` クラスを定義。

### 3.3 損失と正則化
1. `src/losses/task_loss.py`
   - テストグリッドのクロスエントロピーもしくはピクセル精度ベースの損失を実装。
2. `src/losses/share_regularizer.py`
   - SAE の活性化パターンに基づくグループ化、PCA による射影行列計算、平均射影との差分からなる共有正則化を実装。
   - 計算はミニバッチ内の有効グループに限定し、`torch.pca_lowrank` を利用。
3. `src/losses/sae_loss.py`
   - 再構成誤差と L1 を組み合わせた損失を実装し、疎化率に応じた自動重み調整ロジックを追加。
4. `src/train/loss_factory.py`
   - `alpha`, `beta` を含むハイパーパラメータ管理と損失合成ロジックを提供。

### 3.4 トレーニングループ
1. `src/train/trainer.py`
   - 学習ステップ、評価ステップ、チェックポイント保存、Gradient Accumulation、AMP をサポート。
   - 学習率スケジューラ（ウォームアップ＋余弦減衰）と勾配クリップを実装。
2. `src/train/data_module.py`
   - Hydra と連携する `LightningDataModule` 風の実装で、学習・検証・テストタスクの DataLoader を提供。
3. `configs/train.yaml`
   - バッチサイズ、学習率、スケジュール、`alpha`・`beta`、`k` などのハイパーパラメータを定義。
4. ロギング
   - `TensorBoard` または `wandb`（オプション）で損失・精度・活性率を可視化。
   - 学習時間、評価スコア、シード情報を `logs/train/` に JSONL で保存。

### 3.5 評価・メトリクス
1. `src/eval/metrics.py`
   - タスク単位 Top-1/Top-3、ピクセル Accuracy、IoU を実装。
2. `src/eval/ic_evaluator.py`
   - 未見タスクに対する In-Context 推論を実施し、指標を集計するスクリプト。
3. `src/scripts/run_eval.py`
   - チェックポイントを読み込み、`configs/eval.yaml` を参照して評価を実行。
4. 評価結果を `reports/ic_eval/*.json` として保存し、`docs/experiment_log.md` へまとめる。

### 3.6 few-shot メタ適応
1. `src/train/meta_adapter.py`
   - LoRA/Adapter の差し込み、学習率制御、500 step 以内のファインチューニングループを実装。
2. `src/scripts/run_meta_adaptation.py`
   - `data/splits/meta_eval_test.json` を読み込み、各タスクでの適応と性能ログを集計。
3. 指標（到達ステップ数、壁時計時間）を `reports/meta_adaptation/*.json` に保存。

### 3.7 表現解析
1. `src/analysis/cka.py`
   - CKA 計算ユーティリティとタスク間距離の集計。
2. `src/analysis/grassmann.py`
   - Grassmann 距離計算、共有サブスペース指標の算出。
3. `src/analysis/sae_viz.py`
   - SAE 活性ヒートマップ、タスクグルーピング可視化。
4. 結果を `reports/analysis/` 配下に保存し、`docs/findings.md` にサマリを追記。

### 3.8 アブレーション管理
1. `configs/ablations/*.yaml`
   - B0〜B6 各設定ファイルを用意。
2. `src/scripts/run_ablations.py`
   - 設定ごとの一括実行スクリプトを作成し、ログと結果を整理。
3. 主要な差分を `docs/ablation_summary.md` にまとめる。

### 3.9 CI とテスト
1. `tests/` 配下にユニットテスト・スモークテストを追加（データローダー、モデル前向き、正則化計算など）。
2. `pyproject.toml` または `setup.cfg` にテスト依存関係を追記し、`pytest` を標準テストコマンドとする。
3. GitHub Actions もしくは `Makefile` で `pytest`、`ruff`/`flake8`、`mypy` を実行するワークフローを整備。

## 4. マイルストーン
| 期間 | マイルストーン | 主な成果 |
| --- | --- | --- |
| Week 1 | データパイプラインとベースラインモデル（B0）構築 | データ前処理スクリプト、ベースライン学習のスモークテスト |
| Week 2 | 共有正則化・疎AE 統合 | `L_share`, `L_sae` 実装、In-Context 評価スクリプト完成 |
| Week 3 | メタ適応と表現解析 | LoRA/Adapter ループ、CKA/Grassmann/SAE 可視化ツール |
| Week 4 | アブレーションとドキュメント整備 | B0–B6 実験実行、結果サマリと短報ドラフト素材作成 |

## 5. 品質保証と再現性
- すべてのスクリプトで `argparse`/Hydra を用いて設定を明示し、実行時のシードと Git リビジョンをログする。
- 重要なハイパーパラメータは `docs/hyperparameters.md` にまとめ、更新時は履歴を残す。
- 実験結果の可視化ノートブック（`notebooks/`）を準備し、`reports/` に出力画像を保存。

## 6. リスクと緩和策
- **PCA/共有正則化が不安定**：バッチサイズを拡大し、EMA による射影平滑化を導入。必要に応じ `alpha` を段階的に増加。
- **SAE の表現崩壊**：疎性ターゲットを監視し、L1 重みの自動調整と早期停止を併用。
- **学習が収束しない**：ベースライン設定（B0）での安定性を優先し、逐次的に正則化・補助損失を導入。
- **計算資源不足**：グリッド解像度のダウンサンプリング、モデル縮小、LoRA の低ランク化でメモリを節約。

## 7. 今後の更新指針
- 実装進捗に合わせて `docs/experiment_plan.md` と本計画を同期し、重要な仕様変更は両ドキュメントに即時反映する。
- 解析結果に基づく追加タスク（例：解釈可能性向上策）は `docs/backlog.md` を新設して管理する。
