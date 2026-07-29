"""命令列進入點。

    uv run --project D:\\ws\\gt_labeling gt-labeling D:\\ws\\detect_stream\\out\\gt_sample --band 0.5 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .window import MainWindow


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT 標註修正工具")
    parser.add_argument(
        "root", nargs="?", type=Path, help="含 frames/ 與 labels/ 的資料夾;省略則開啟選擇對話框"
    )
    parser.add_argument(
        "--band",
        nargs=2,
        type=float,
        metavar=("Y1", "Y2"),
        help="偵測帶上下界(歸一化 y,0~1),畫成水平參考線;帶外會壓暗提醒不要標",
    )
    parser.add_argument("--cache", type=int, default=8, help="解碼後 QPixmap 的快取張數(預設 8)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for value in args.band or ():
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--band 必須是歸一化 [0,1] 的 y 值,收到 {value}")

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("asys")
    app.setApplicationName("gt_labeling")

    window = MainWindow(cache_size=args.cache)
    if args.band:
        window.set_band(*args.band)
    window.show()
    if args.root is not None:
        window.open_root(args.root)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
