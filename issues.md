- ARC-AGI-2データはtraining, evaluationが分かれているのに、データ前処理でk-shot対応の分割をする前提になっている
  - データ分割前提であるため、run_eval.pyやrun_eval_with_visualization.pyで分割していないevaluationデータを使うことができない
  - 今のスクリプトでevaluationデータを処理できるようにするため、data/processedのデータについて、以下のように処理できる前処理スクリプトが必要
    - data/processed_training-k-shot
    - data/processed_evaluation
    - data/processed_test
    