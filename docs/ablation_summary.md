# アブレーション実験サマリ

本ドキュメントは `configs/ablations/` に整理した B0〜B6 の設定内容と、`src/scripts/run_ablations.py` による一括実行手順をまとめる。各設定はベースライン学習に対して特定の要素を切り替え、共有サブスペース正則化および疎オートエンコーダの寄与を評価するために設計している。

## 実行手順
1. 必要な前処理（`data/processed_training-k-shot/`, `data/splits/` の生成）と学習チェックポイントの配置を完了させる。
2. 任意のシェルから以下のコマンドを実行する。
   ```bash
   python -m src.scripts.run_ablations
   ```
   - 既定では `configs/ablations/` 配下の YAML を全て読み込み、結果を `reports/ablations/summary.json` に集約する。
   - `--only B0 B2` のように指定すると対象アブレーションを絞り込める。
   - 各ステージ（`train`, `meta_adaptation`）の詳細結果は YAML 内で指定した `reports/...` に JSON として保存される。

## 各設定の概要
| ID | 目的 | 主な変更点 | 主な出力 |
| --- | --- | --- | --- |
| **B0** | 正則化なしのベースライン | `alpha=0`, `beta=0` とし SAE 損失を完全に無効化。 | `reports/ablations/B0.json` |
| **B1** | SAE のみ | 共有正則化を除外し、`beta=0.1` で SAE 再構成＋疎性を維持。 | `reports/ablations/B1.json` |
| **B2** | 共有のみ | `beta=0` とし共有サブスペース正則化のみを残す。閾値とグループ最小サイズを調整。 | `reports/ablations/B2.json` |
| **B3** | SAE 非疎化 | `sae_l1_weight=0` として疎性制約を解除、再構成項のみで SAE を駆動。 | `reports/ablations/B3.json` |
| **B4** | メタ適応でほぼ全層更新 | 学習バッチサイズを調整しつつ、メタ適応ではボトルネック 256 の Adapter を挿入して全層更新を近似。 | `reports/ablations/B4.json`, `reports/meta_adaptation/ablations/B4.json` |
| **B5** | コンテキストエンコーダ比較 | Transformer 構造の層数・ヘッド数・次元を変化させて擬似的に別アーキテクチャを再現。 | `reports/ablations/B5_*.json` |
| **B6** | 共有ランク／αスイープ | `share_rank` と `alpha` の組を (8,0.05) / (16,0.1) / (32,0.2) で比較。 | `reports/ablations/B6_*.json` |

### B5 のバリアント
- `B5_transformer`: 既定の Transformer アーキテクチャをそのまま利用するリファレンス。
- `B5_lightweight`: 層数 2、ヘッド数 4、`dropout=0.2` で軽量化した構成。
- `B5_single_head`: `context_heads=1`、`context_model_dim=192` に設定し、疑似的な GRU スタイルの低容量エンコーダを再現。

### B6 のバリアント
- `k8_alpha005`: `share_rank=8`, `alpha=0.05`。最も弱い共有制約。
- `k16_alpha01`: `share_rank=16`, `alpha=0.1`。基準より強い共有化。
- `k32_alpha02`: `share_rank=32`, `alpha=0.2`。最大ランクで最も強い正則化。

## 注意事項
- B4 のメタ適応は `adapter.mode=adapter` かつボトルネック次元を大きく取ることで「全層微調整に近い」状態を再現している。厳密な全層更新ではない点に留意すること。
- 大規模なランク設定（例: B6 の `share_rank=32`）は GPU メモリ使用量を増加させるため、必要に応じてバッチサイズや AMP 設定を調整する。
- 生成されたサマリ JSON は再実行時に上書きされる。履歴を保持したい場合は別名で退避させること。
