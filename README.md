# ProtoMotions（G1 / AMP 拡張フォーク）

物理シミュレーション上のヒューマノイドに、モーションキャプチャから「歩き方・跳び方」などの動きの分布を学習させる、模倣学習＋強化学習の研究・実験コードです。

本リポジトリは [NVLabs/ProtoMotions](https://github.com/NVLabs/ProtoMotions) をベースに、**Unitree G1** への適用と **Adversarial Motion Priors (AMP)** の再現・調整を中心に拡張しています。

> **注（採用担当者向け）**  
> フレームワーク本体（マルチシミュレータ抽象化、PPO、Hydra 構成、MaskedMimic 等）は上流プロジェクトの成果です。本フォークで実装・検証したのは、G1 向けアセット／リターゲティング、AMP の損失・報酬の原論文準拠化、実験設定の追加です。背景・目的の段落は仮置きなので、応募時に実績に合わせて書き換えてください。

---

## 開発の背景・目的（仮）

ヒューマノイドの運動制御では、タスク報酬だけだと「目的は達成するが動きが不自然」になりやすいです。AMP は、参照モーションの分布に近いスタイルを識別器で学習し、方策の報酬に乗せることで、自然な歩行・アクロバット動作を獲得する手法です。

本プロジェクトでは次を目的としました（仮）。

1. 研究用キャラクタ（SMPL / AMP humanoid）ではなく、実機に近い **Unitree G1** で AMP を動かす。
2. BVH / AMASS から G1 関節への **リターゲティング** を通し、バックフリップ等の難易度の高い動作を学習データにする。
3. 実装が論文・原版とずれていた AMP の **識別器損失・スタイル報酬** を原版に寄せ、学習の安定性を検証する。

---

## 使用技術

| 領域 | 技術 | バージョン目安 |
|------|------|----------------|
| 言語 | Python | 3.10 推奨（Genesis 公式要件に合わせる場合） |
| 深層学習 | PyTorch | IsaacGym: `2.2` / IsaacLab・Genesis: `2.5.0` 系 |
| 学習ループ | PyTorch Lightning Fabric | IsaacGym: `2.3.3` / IsaacLab: `2.5.0.post0` |
| 設定 | Hydra / OmegaConf | `hydra-core` 1.2〜1.3 |
| アルゴリズム | PPO、AMP、ASE、DeepMimic 系、MaskedMimic | 上流実装＋本フォークの AMP 調整 |
| シミュレータ | NVIDIA Isaac Gym / Isaac Lab、Genesis | いずれか 1 つを選択 |
| 実験管理 | Weights & Biases | `wandb` 0.19 系 |
| 人体モデル・リターゲット | SMPL/SMPL-X、PoseLib、[Mink](https://github.com/kevinzakka/mink) | — |
| ロボット | Unitree G1（手・手首あり構成を追加） | URDF / MJCF / USD |
| パッケージ | `setup.py` 上の dist 名 `protomotions` | `2.0` |

シミュレータごとのピン留めは `requirements_isaacgym.txt` / `requirements_isaaclab.txt` / `requirements_genesis.txt` を参照してください。

---

## 主な機能・本フォークで実装したこと

### 上流（ProtoMotions）が提供するもの

- シミュレータ非依存の環境・エージェント構成（IsaacGym / IsaacLab / Genesis）
- PPO を核とした学習（`protomotions/train_agent.py`）と評価（`protomotions/eval_agent.py`）
- AMP / ASE / 全身トラッキング（DeepMimic 拡張）/ MaskedMimic
- 地形・シーン生成、モーションライブラリ、Hydra による実験合成

### 本フォークで追加・変更したもの

- **Unitree G1 のロボット定義**  
  `g1` / `g1_hand` / `g1_hand_wrist`（関節名・PD ゲイン・観測次元・URDF/USD/MJCF）。
- **BVH → Isaac（PoseLib）変換**  
  `data/scripts/convert_bvh_to_isaac.py`。オイラー角からクォータニオンへの変換で時間方向の符号連続性を保つ処理を実装。
- **Mink リターゲティングの G1 対応**  
  `data/scripts/retargeting/mink_retarget.py` にキーポイント対応・速度制限・MJCF パスを追加。AMASS 変換（`convert_amass_to_isaac.py`）からも G1 系を指定可能。
- **学習用モーション**  
  歩行系クリップに加え、バックフリップ等を G1（手・手首）スケルトンへマッピングした `.npy` を追加。
- **AMP の原版寄せ**  
  識別器を Least-Squares GAN 系の損失・報酬に変更し、weight decay や replay buffer サイズなどのハイパーパラメータを調整（コミット時点）。作業ツリーでは論文の logistic 損失への切り替えも残しています。
- **実験設定**  
  `amp_mlp` の整理、`amp_transformer`（Transformer actor + 地形対応）、`deepmimic_mlp` の観測設定。
- **履歴観測**  
  AMP 識別器入力（状態の時系列 `s, s', …`）向けに `HumanoidObs` の履歴リセット・デモ観測を拡張。

---

## こだわったポイント

- **論文実装との対応**  
  AMP は「識別器の目的関数」と「方策に渡すスタイル報酬」がずれると学習が壊れやすいです。BCE（logistic）と LS-GAN をコード上で切り替えられる形にし、原論文・原版コードの数式に戻せるようにしています。勾配ペナルティ・replay 混合・weight decay も識別器側に残しています。
- **実機骨格へのリターゲティング**  
  SMPL 空間のモーションを G1 の DoF・ボディ名に落とす際、固定関節の扱い、左右反転、Mink のキーポイント対応をロボット種別で分岐しています。BVH ではクォータニオンの符号ジャンプ（\(q\) と \(-q\)）が IK・補間を壊すため、時間方向に符号を揃えています。
- **実験の再現性**  
  学習エントリは Hydra の composition（`+exp` / `+robot` / `+simulator`）に寄せ、チェックポイントディレクトリの `config.yaml` から評価時に設定を復元します。
- **シミュレータ差の吸収**  
  IsaacGym は `torch` より先に import する必要があるため、CLI を先に走査してから Fabric / 環境を起動しています。

改善余地: `convert_bvh_to_isaac.py` にデバッグ用の `ipdb` が残っています。

---

## リポジトリ構成（抜粋）

```
protomotions/
  train_agent.py          # 学習エントリ（Hydra + Fabric）
  eval_agent.py           # 評価・可視化
  agents/                 # PPO / AMP / ASE / Mimic / MaskedMimic
  envs/                   # タスク・観測コンポーネント
  simulator/              # IsaacGym / IsaacLab / Genesis
  config/                 # Hydra 設定
data/scripts/             # AMASS/BVH 変換・リターゲティング
poselib/                  # スケルトン・回転ユーティリティ
isaac_utils/
```

---

## ローカルでの環境構築・起動手順

GPU 付き Linux を前提とします。まず LFS を取得してください。

```bash
git lfs fetch --all
```

シミュレータは **どれか 1 つ** を入れます。詳細な既知の制限は上流 README と同様です（Genesis のシーン／カプセル人体、IsaacLab の一部キャラクタ未検証など）。

### 1. IsaacGym

1. [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) を導入し、Python API を入れる。
2. リポジトリルートで:

```bash
pip install -e .
pip install -r requirements_isaacgym.txt
pip install -e isaac_utils
pip install -e poselib
alias PYTHON_PATH=python
```

メモリ不足時は `num_envs=1024` のように環境数を下げてください。`python` 関連のリンクエラーでは `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/` を試します。

### 2. IsaacLab

1. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) を導入する。
2. `isaaclab.sh` 経由で Python を使う。

```bash
alias PYTHON_PATH="/path/to/IsaacLab/isaaclab.sh -p"
PYTHON_PATH -m pip install -e .
PYTHON_PATH -m pip install -r requirements_isaaclab.txt
PYTHON_PATH -m pip install -e isaac_utils
PYTHON_PATH -m pip install -e poselib
```

### 3. Genesis

Python 3.10 で [Genesis](https://genesis-world.readthedocs.io/en/latest/index.html) を入れたうえで:

```bash
pip install -e .
pip install -r requirements_genesis.txt
pip install -e isaac_utils
pip install -e poselib
alias PYTHON_PATH=python
```

### 学習（例）

コマンドは `+robot` を `+simulator` より先に指定してください（IsaacGym の import 順のため）。

```bash
# 上流の例: H1 のステアリング（純粋なタスク報酬 + PPO）
PYTHON_PATH protomotions/train_agent.py \
  +exp=steering_mlp +robot=h1 +simulator=isaacgym +experiment_name=h1_steering

# 本フォークの主眼: G1 + AMP（モーションファイルを指定）
PYTHON_PATH protomotions/train_agent.py \
  +exp=amp_mlp +robot=g1_hand_wrist +simulator=isaacgym \
  motion_file=<path-to-npy-or-yaml> +experiment_name=g1_amp_mlp

# Transformer actor 版
PYTHON_PATH protomotions/train_agent.py \
  +exp=amp_transformer +robot=g1_hand_wrist +simulator=isaacgym \
  motion_file=<path-to-npy-or-yaml> +experiment_name=g1_amp_transformer
```

`experiment_name` が同じだと `results/<name>/last.ckpt` から自動再開します。ログは `+opt=wandb` を付けると W&B に送れます。

### 評価

```bash
PYTHON_PATH protomotions/eval_agent.py \
  +robot=g1_hand_wrist +simulator=isaacgym \
  motion_file=<path-to-motion> \
  checkpoint=results/<experiment_name>/last.ckpt
```

| キー | 動作 |
|------|------|
| `J` | 外力を加えて頑健性を確認 |
| `R` | タスクリセット |
| `O` | カメラ対象の切り替え |
| `L` | 録画開始／保存 |
| `;` | 録画キャンセル |
| `Q` | 終了 |

モーションのキネマティック再生:

```bash
PYTHON_PATH protomotions/scripts/play_motion.py <motion_file> <isaacgym|isaaclab|genesis> <robot_type>
```

### データ変換（G1 向け・概要）

1. AMASS / BVH を用意する。SMPL パラメータは上流ドキュメントどおり `data/smpl/` に配置。
2. AMASS: `python data/scripts/convert_amass_to_isaac.py <AMASS_dir> --robot-type=g1_hand_wrist --force-retarget`
3. BVH: `python data/scripts/convert_bvh_to_isaac.py`（`robot_type` に `g1` / `g1_hand` / `g1_hand_wrist`）

ライセンス上公開できないモーションはリポジトリに含めず、パスだけ README に書いてください。

---

## 引用・ライセンス

上流および依存プロジェクトのライセンスに従ってください。学術利用時は ProtoMotions / MaskedMimic / AMP / ASE 等の原論文の引用を推奨します。BibTeX は [NVLabs/ProtoMotions](https://github.com/NVLabs/ProtoMotions) を参照してください。

```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions: Physics-based Character Animation},
  author = {Tessler, Chen and Juravsky, Jordan and Guo, Yunrong and Jiang, Yifeng and Coumans, Erwin and Peng, Xue Bin},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/NVLabs/ProtoMotions/}},
}
```
