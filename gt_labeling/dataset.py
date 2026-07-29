"""資料夾掃描與影像快取。

* frames/ 與 labels/ 以 **stem 對應**;JSON 裡的 ``image`` 欄位只原樣保留、不參與定位
  (它是 ``../frames/...`` 相對路徑,用它定位會綁死目錄結構)。
* 標註檔在開啟時一次全部載入記憶體(75 幀 x ~1KB,可忽略)。好處:清單能立刻顯示
  框數與待補狀態,且關掉自動存時未存編輯能跨幀保留。
* 3840x1920 JPEG 解碼約數十毫秒,**解碼結果以 LRU 快取 QPixmap**,重繪不重新解碼;
  切幀後在背景執行緒預先解碼前後鄰幀(QImage 可跨執行緒,QPixmap 只能在 GUI 執行緒建立)。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from .model import FrameLabel, load_frame

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True, slots=True)
class FrameEntry:
    seq: int
    stem: str
    label_path: Path
    image_path: Path | None


def scan_root(root: Path) -> list[FrameEntry]:
    root = Path(root)
    labels_dir = root / "labels"
    frames_dir = root / "frames"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"找不到 labels 資料夾:{labels_dir}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"找不到 frames 資料夾:{frames_dir}")

    entries: list[FrameEntry] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        image_path = next(
            (p for ext in IMAGE_EXTS if (p := frames_dir / f"{stem}{ext}").is_file()), None
        )
        entries.append(
            FrameEntry(seq=_seq_of(stem, label_path), stem=stem,
                       label_path=label_path, image_path=image_path)
        )
    entries.sort(key=lambda e: (e.seq, e.stem))
    return entries


def _seq_of(stem: str, label_path: Path) -> int:
    """檔名是數字就用檔名(最省),否則退回讀 JSON 的 seq 欄位。"""
    if stem.isdigit():
        return int(stem)
    try:
        return int(json.loads(label_path.read_text(encoding="utf-8-sig")).get("seq", 0))
    except (OSError, ValueError, TypeError):
        return 0


def load_all(entries: list[FrameEntry]) -> list[FrameLabel]:
    return [load_frame(e.label_path) for e in entries]


class _Decoded(QObject):
    delivered = pyqtSignal(int, QImage)


class _DecodeTask(QRunnable):
    def __init__(self, seq: int, path: Path, sink: _Decoded) -> None:
        super().__init__()
        self._seq = seq
        self._path = path
        self._sink = sink

    def run(self) -> None:  # 背景執行緒
        image = QImage(str(self._path))
        self._sink.delivered.emit(self._seq, image)


class ImageStore(QObject):
    """seq -> QPixmap 的 LRU 快取,附背景預解碼。"""

    ready = pyqtSignal(int)

    def __init__(self, capacity: int = 8, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.capacity = max(2, capacity)
        self._cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._pending: set[int] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._sink = _Decoded(self)
        self._sink.delivered.connect(self._on_decoded)  # 佇列到 GUI 執行緒

    def clear(self) -> None:
        self._cache.clear()
        self._pending.clear()

    def peek(self, seq: int) -> QPixmap | None:
        pixmap = self._cache.get(seq)
        if pixmap is not None:
            self._cache.move_to_end(seq)
        return pixmap

    def load_now(self, entry: FrameEntry) -> QPixmap | None:
        """同步取得當前幀:命中快取就不重新解碼。"""
        cached = self.peek(entry.seq)
        if cached is not None:
            return cached
        if entry.image_path is None:
            return None
        pixmap = QPixmap(str(entry.image_path))
        if pixmap.isNull():
            return None
        self._insert(entry.seq, pixmap)
        return pixmap

    def prefetch(self, entries: list[FrameEntry]) -> None:
        for entry in entries:
            if entry.image_path is None or entry.seq in self._cache or entry.seq in self._pending:
                continue
            self._pending.add(entry.seq)
            self._pool.start(_DecodeTask(entry.seq, entry.image_path, self._sink))

    def _on_decoded(self, seq: int, image: QImage) -> None:  # GUI 執行緒
        self._pending.discard(seq)
        if image.isNull() or seq in self._cache:
            return
        self._insert(seq, QPixmap.fromImage(image))
        self.ready.emit(seq)

    def _insert(self, seq: int, pixmap: QPixmap) -> None:
        self._cache[seq] = pixmap
        self._cache.move_to_end(seq)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
