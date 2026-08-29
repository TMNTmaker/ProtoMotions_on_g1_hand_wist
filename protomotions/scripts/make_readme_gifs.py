"""README に貼る GIF を、評価時の録画やスクリーンキャストから作る。

GitHub の README では mp4 / webm をインライン再生できないため、成果物は
容量を抑えた GIF に変換して assets/ に置く。どのクリップをどう切り出したかを
再現できるように、切り出し範囲と切り抜き位置は CLIPS に定義してある。

使い方:
    /isaac-sim/python.sh protomotions/scripts/make_readme_gifs.py
    /isaac-sim/python.sh protomotions/scripts/make_readme_gifs.py --only walk
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import typer


@dataclass
class Clip:
    source: str
    start: float
    duration: float
    # Isaac Sim の UI を落とすための切り抜き（ffmpeg の crop=w:h:x:y と同じ）
    crop: Optional[str] = None
    # README の読み込みが重くならないよう 1 本 3 MB 前後に収まる設定にしている。
    width: int = 500
    fps: int = 10


# 出力名 -> 切り出し設定
CLIPS: Dict[str, Clip] = {
    "walk_steering": Clip(
        source="movies/walk.webm",
        start=44.0,
        duration=10.0,
        crop="1360:850:20:35",
    ),
    # 評価時の録画は視点がロボットに追従するので、切り抜き無しでそのまま使える。
    "run_transformer": Clip(
        source="movies/run_amp_transformer.mp4",
        start=0.0,
        duration=7.0,
    ),
    # 引きの画で地面のテクスチャが毎フレーム変わるぶんデータ量が増えるので、
    # 他より解像度を落としている。
    "strange_run": Clip(
        source="movies/strange_run.webm",
        start=4.0,
        duration=9.0,
        crop="850:640:240:60",
        width=440,
    ),
    "backflip_failure": Clip(
        source="movies/faild_backflips_amp_transformer.mp4",
        start=0.0,
        duration=4.5,
    ),
    "backflip_training": Clip(
        source="movies/faild_backflips_amp_transformer_training.webm",
        start=30.0,
        duration=10.0,
        crop="1200:610:150:131",
    ),
}


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def convert(clip: Clip, dst: Path) -> None:
    filters = [f"fps={clip.fps}"]
    if clip.crop:
        filters.insert(0, f"crop={clip.crop}")
    filters.append(f"scale={clip.width}:-1:flags=lanczos")
    chain = ",".join(filters)

    trim = ["-ss", str(clip.start), "-t", str(clip.duration)]
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        # 1 パスで減色すると色が破綻するため palettegen / paletteuse の 2 パスにする。
        subprocess.run(
            [ffmpeg_exe(), "-y", *trim, "-i", clip.source,
             "-vf", f"{chain},palettegen=stats_mode=diff", str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ffmpeg_exe(), "-y", *trim, "-i", clip.source, "-i", str(palette),
             "-lavfi", f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", str(dst)],
            check=True, capture_output=True,
        )


def main(
    out_dir: Path = Path("assets"),
    only: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, clip in CLIPS.items():
        if only and only != name:
            continue
        if not Path(clip.source).exists():
            print(f"元動画が見つからない: {clip.source}")
            continue
        dst = out_dir / f"{name}.gif"
        print(f"変換: {clip.source} [{clip.start}s +{clip.duration}s] -> {dst}")
        convert(clip, dst)
        print(f"  {dst.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    typer.run(main)
