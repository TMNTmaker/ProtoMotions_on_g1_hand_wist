"""results/ 配下の TensorBoard ログから README 用の学習曲線を描く。

このリポジトリのロガーはスカラーを step=0 で書き出すため、横軸には
info/frames（累積環境ステップ数）の値を使う。学習を再開した実験は
version_* が複数に分かれるので、frames の値で並べ直して 1 本に繋げている。

tfevents の読み出しに数分かかるので、抽出結果を JSON にキャッシュしている。

使い方:
    /isaac-sim/python.sh protomotions/scripts/plot_training_curves.py
    /isaac-sim/python.sh protomotions/scripts/plot_training_curves.py --refresh
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

# 図に出す実験。値は results/ 配下のディレクトリ名。
DEFAULT_RUNS: Dict[str, str] = {
    "run / MLP": "g1_run_amp",
    "run / Transformer": "g1_run_amp_transformer",
    "backflip / MLP": "g1_backflip_amp",
    "backflip / Transformer": "g1_backflip_amp_transformer",
}

COLORS = {
    "run / MLP": "#1f77b4",
    "run / Transformer": "#ff7f0e",
    "backflip / MLP": "#2ca02c",
    "backflip / Transformer": "#d62728",
}

X_TAG = "info/frames"
LENGTH_TAG = "info/episode_length"

# (タグ, 縦軸ラベル)。survival と logit_margin は生タグから作る派生量。
# この 4 実験は転倒による早期終了を入れていないため、生存率と早期終了率は
# 全実験で平坦になる。図には出さず、値の確認はサマリ表で行う。
CURVE_PANELS: List[Tuple[str, str]] = [
    ("rewards/amp_rewards", "AMP style reward"),
    ("discriminator/logit_margin", "Logit margin (demo - policy)"),
]
DISCRIMINATOR_PANELS: List[Tuple[str, str]] = [
    ("discriminator/agent_acc", "Accuracy on policy samples"),
    ("discriminator/grad_penalty", "Gradient penalty"),
]
# 図にはしないが README の記述根拠として値を確認したいタグ
SUMMARY_EXTRA_TAGS: List[str] = [
    "survival",
    "env/terminate_frac",
    LENGTH_TAG,
    "discriminator/pos_acc",
]

# tfevents から読み出す実タグ（派生量の材料を含む）
SOURCE_TAGS = [
    X_TAG,
    LENGTH_TAG,
    "rewards/amp_rewards",
    "env/terminate_frac",
    "discriminator/agent_acc",
    "discriminator/pos_acc",
    "discriminator/agent_logit_mean",
    "discriminator/expert_logit_mean",
    "discriminator/grad_penalty",
]


def read_max_episode_length(exp_dir: Path) -> Optional[float]:
    """学習時の config.yaml から 1 エピソードの上限ステップ数を読む。"""
    import yaml

    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text())
    try:
        return float(config["env"]["config"]["max_episode_length"])
    except (KeyError, TypeError, ValueError):
        return None


def extract(log_dir: Path, max_points: int) -> Dict[str, List[float]]:
    """1 つの version ディレクトリから必要なスカラー列を取り出す。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # size_guidance を絞らないと数十 MB のログで待たされる。全タグが毎エポック
    # 同じ回数記録され、間引きの乱数種も共通なので、抽出後も index は揃う。
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": max_points})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])

    series: Dict[str, List[float]] = {}
    for tag in SOURCE_TAGS:
        if tag in available:
            series[tag] = [event.value for event in accumulator.Scalars(tag)]
    return series


def extract_run(exp_dir: Path, max_points: int) -> Dict[str, List[float]]:
    """学習再開で分かれた version_* を frames 順に繋げて 1 本の系列にする。"""
    versions = sorted((exp_dir / "lightning_logs").glob("version_*"))
    chunks = []
    for version in versions:
        if not list(version.glob("*tfevents*")):
            continue
        series = extract(version, max_points)
        if X_TAG not in series:
            continue
        lengths = {len(values) for values in series.values()}
        if len(lengths) != 1:
            print(f"  {version.name}: タグ間で長さが不一致 {lengths} のためスキップ")
            continue
        print(f"  {version.name}: {len(series[X_TAG])} 点")
        chunks.append(series)

    if not chunks:
        return {}

    tags = set(chunks[0])
    for chunk in chunks[1:]:
        tags &= set(chunk)

    merged = {tag: [] for tag in tags}
    for chunk in chunks:
        for tag in tags:
            merged[tag].extend(chunk[tag])

    order = sorted(range(len(merged[X_TAG])), key=lambda i: merged[X_TAG][i])
    return {tag: [values[i] for i in order] for tag, values in merged.items()}


def load_all(
    results_dir: Path, runs: Dict[str, str], cache_path: Path, max_points: int
) -> Dict[str, Dict]:
    if cache_path.exists():
        print(f"キャッシュを利用: {cache_path}")
        return json.loads(cache_path.read_text())

    data: Dict[str, Dict] = {}
    for label, name in runs.items():
        exp_dir = results_dir / name
        if not exp_dir.exists():
            print(f"スキップ（存在しない）: {exp_dir}")
            continue
        print(f"読み込み中: {label} <- {exp_dir}")
        series = extract_run(exp_dir, max_points)
        if not series:
            continue
        data[label] = {
            "series": series,
            "max_episode_length": read_max_episode_length(exp_dir),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    print(f"キャッシュを書き出し: {cache_path}")
    return data


def derive(run: Dict, tag: str):
    """派生量を含めてタグの生の値列を返す。"""
    import numpy as np

    series = run["series"]
    if tag == "survival":
        if LENGTH_TAG not in series or not run.get("max_episode_length"):
            return None
        return np.asarray(series[LENGTH_TAG]) / run["max_episode_length"]
    if tag == "discriminator/logit_margin":
        if not {"discriminator/expert_logit_mean", "discriminator/agent_logit_mean"} <= set(series):
            return None
        return np.asarray(series["discriminator/expert_logit_mean"]) - np.asarray(
            series["discriminator/agent_logit_mean"]
        )
    if tag not in series:
        return None
    return np.asarray(series[tag], dtype=float)


def to_xy(run: Dict, tag: str, smooth: float):
    """(環境ステップ数[M], 平滑化した値) を返す。"""
    import numpy as np

    values = derive(run, tag)
    if values is None:
        return None, None
    x = np.asarray(run["series"][X_TAG], dtype=float) / 1e6

    if smooth > 0:
        smoothed = np.empty_like(values, dtype=float)
        acc = float(values[0])
        for i, value in enumerate(values):
            acc = smooth * acc + (1.0 - smooth) * float(value)
            smoothed[i] = acc
        values = smoothed
    return x, values


def draw(
    data: Dict[str, Dict],
    panels: List[Tuple[str, str]],
    out_path: Path,
    smooth: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0))
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (tag, ylabel) in zip(axes, panels):
        for label, run in data.items():
            x, y = to_xy(run, tag, smooth)
            if x is None:
                continue
            ax.plot(x, y, label=label, color=COLORS.get(label), linewidth=1.3)
        ax.set_xlabel("Environment steps [M]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"保存: {out_path}")


def summarize(data: Dict[str, Dict], smooth: float) -> None:
    """README に載せる数値を確認するためのサマリ。"""
    import numpy as np

    tags = [tag for tag, _ in CURVE_PANELS + DISCRIMINATOR_PANELS] + SUMMARY_EXTRA_TAGS
    header = f"{'run':<26}{'steps[M]':>10}" + "".join(
        f"{tag.split('/')[-1]:>18}" for tag in tags
    )
    print("\n" + header)
    print("-" * len(header))
    for label, run in data.items():
        x, _ = to_xy(run, CURVE_PANELS[0][0], 0.0)
        row = f"{label:<26}{(x[-1] if x is not None else 0):>10.0f}"
        for tag in tags:
            _, y = to_xy(run, tag, smooth)
            row += f"{'--':>18}" if y is None else f"{np.mean(y[-20:]):>18.3f}"
        print(row)


def main(
    results_dir: Path = Path("results"),
    out_dir: Path = Path("assets"),
    cache: Path = Path("output/training_curves_cache.json"),
    max_points: int = 2000,
    smooth: float = 0.9,
    refresh: bool = False,
    runs_json: Optional[Path] = None,
) -> None:
    runs = json.loads(runs_json.read_text()) if runs_json else DEFAULT_RUNS
    if refresh and cache.exists():
        cache.unlink()

    data = load_all(results_dir, runs, cache, max_points)
    if not data:
        raise typer.Exit("読み込めるログがありませんでした")

    draw(
        data,
        CURVE_PANELS,
        out_dir / "training_curves.png",
        smooth,
        "G1 AMP training curves",
    )
    draw(
        data,
        DISCRIMINATOR_PANELS,
        out_dir / "discriminator_balance.png",
        smooth,
        "AMP discriminator behaviour",
    )
    summarize(data, smooth)


if __name__ == "__main__":
    typer.run(main)
