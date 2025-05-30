"""参照を多段に持つ USD をフラット化して 1 ファイルに書き出す。

URDF Importer が出力する USD は sublayer 参照を含み、IsaacLab から読むと
参照解決に失敗することがあるため、事前にフラット化しておく。
"""

import sys

from pxr import Usd


def flatten(src_path: str, dst_path: str) -> None:
    stage = Usd.Stage.Open(src_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {src_path}")
    stage.Flatten().Export(dst_path)
    print(f"Exported flattened stage to {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: usd_flat.py <src.usd> <dst.usd>")
    flatten(sys.argv[1], sys.argv[2])
