"""GT JSON 的資料結構與讀寫。

存檔契約(驗收依據):

* **只替換 ``raw["dets"]``**。``type / version / source / seq / frame_index /
  video_sec / size / image`` 連 key 順序都是原本那顆 dict,不重建。
* **det 也一樣**:只覆寫 ``label / track_id / ppe / bbox``,其餘欄位連 key 順序
  原樣帶回。上游 ``gt_densify.py`` 會寫 ``src``(anchor / det / interp),
  ``eval_gt.py`` 靠它算「多少比例的框取自系統輸出」——剝掉不會讓任何流程報錯,
  只會讓那個比例悄悄低報,所以這條是硬性契約。
* 行尾(CRLF/LF)、UTF-8 BOM、結尾換行都照原檔還原,所以 dets 沒改時存回去
  是 **byte 完全相同**,不只是欄位值相同。
* ``bbox`` 一律寫成 5 位小數、clamp 到 ``[0,1]``、保證 ``x1<x2`` / ``y1<y2``。
* 存檔後把記憶體的 bbox 回填成寫出去的值,所以「存檔 -> 繼續編輯 -> 再存」與
  「存檔 -> 重開」看到的座標完全一致(round 到 5 位是冪等的)。
"""

from __future__ import annotations

import codecs
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LABELS: tuple[str, ...] = ("person", "drone")
PPE_VALUES: tuple[str, ...] = ("ok", "ng")
BBOX_DP = 5
MIN_SPAN = 10.0**-BBOX_DP


def canonical_bbox(bbox) -> list[float]:
    """排序 -> clamp [0,1] -> round 5 位 -> 保證有非零寬高。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    x1, y1, x2, y2 = (round(v, BBOX_DP) for v in (x1, y1, x2, y2))
    x1, x2 = _ensure_span(x1, x2)
    y1, y2 = _ensure_span(y1, y2)
    return [x1, y1, x2, y2]


def _ensure_span(lo: float, hi: float) -> tuple[float, float]:
    if hi - lo >= MIN_SPAN:
        return lo, hi
    if lo + MIN_SPAN <= 1.0:
        return lo, round(lo + MIN_SPAN, BBOX_DP)
    return round(hi - MIN_SPAN, BBOX_DP), hi


@dataclass(slots=True)
class Det:
    label: str = "person"
    track_id: int | None = None
    ppe: str | None = None
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    # 讀進來的原始 det,**視為唯讀**:存檔時當底、只覆寫本工具負責的四個欄位,
    # 產生它的上游(gt_densify 的 src 等)寫了什麼就原樣還回去。
    raw: dict = field(default_factory=dict, repr=False)

    def clone(self) -> Det:
        # raw 共享參考而非複製:它從不被修改,而 clone 走在 undo 快照的熱路徑上。
        return Det(self.label, self.track_id, self.ppe, list(self.bbox), self.raw)

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    @property
    def pending(self) -> bool:
        """待補:track_id 未指定,或 person 的 ppe 未判定。"""
        return self.track_id is None or (self.is_person and self.ppe is None)

    def to_json(self) -> dict:
        # 以原始 det 當底再覆寫:更新既有 key 不會改變 dict 的順序,所以連
        # key 排列都跟讀進來時一樣;上游多寫的欄位(src 等)也原封不動帶回去。
        out = dict(self.raw)
        out["label"] = self.label
        out["track_id"] = self.track_id
        out["ppe"] = self.ppe if self.is_person else None
        out["bbox"] = canonical_bbox(self.bbox)
        return out

    def display_text(self) -> str:
        parts = [self.label, f"#{self.track_id}" if self.track_id is not None else "#?"]
        if self.is_person:
            parts.append(self.ppe or "?")
        return " ".join(parts)


@dataclass(slots=True)
class TextStyle:
    """原檔的位元組層外觀,存檔時原樣還原。"""

    newline: str = "\r\n" if os.name == "nt" else "\n"
    trailing_newline: bool = False
    bom: bool = False

    def render(self, body: str) -> bytes:
        if self.newline != "\n":
            body = body.replace("\n", self.newline)
        if self.trailing_newline:
            body += self.newline
        return (codecs.BOM_UTF8 if self.bom else b"") + body.encode("utf-8")


@dataclass
class FrameLabel:
    path: Path
    raw: dict
    dets: list[Det]
    style: TextStyle
    _clean: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ 唯讀資訊

    @property
    def seq(self) -> int:
        return int(self.raw.get("seq", 0))

    @property
    def size(self) -> tuple[int, int]:
        raw_size = self.raw.get("size")
        if not (isinstance(raw_size, (list, tuple)) and len(raw_size) == 2):
            raise ValueError(f"{self.path.name} 缺少合法的 size 欄位,無法換算 normalized 座標")
        return int(raw_size[0]), int(raw_size[1])

    @property
    def dirty(self) -> bool:
        return self.dets_json() != self._clean

    @property
    def pending_count(self) -> int:
        return sum(1 for d in self.dets if d.pending)

    @property
    def has_null_track(self) -> bool:
        return any(d.track_id is None for d in self.dets)

    @property
    def has_null_ppe(self) -> bool:
        return any(d.is_person and d.ppe is None for d in self.dets)

    @property
    def has_duplicate_track(self) -> bool:
        """同一幀有兩個框共用同一個 ``(label, track_id)``。

        一個目標在一幀裡只該有一個框,重複多半是換號撞到既有號碼、或上游把兩個
        目標配成同一條軌跡,兩種都得回頭看。

        跨 label 同號不算重複:軌跡身分是 ``(label, track_id)``,person#1 與
        drone#1 本來就是兩條軌跡,而兩種 label 常各自從 0 開始編號——算進來的話
        警示會每幀都亮,等於沒有警示。
        """
        seen = set()
        for det in self.dets:
            if det.track_id is None:
                continue
            key = (det.label, det.track_id)
            if key in seen:
                return True
            seen.add(key)
        return False

    # ------------------------------------------------------------------ 序列化

    def dets_json(self) -> list[dict]:
        return [d.to_json() for d in self.dets]

    def save(self, force: bool = False) -> bool:
        """寫回原 JSON。沒改動就不動檔案(回傳 False)。"""
        payload = self.dets_json()
        if not force and payload == self._clean:
            return False

        self.raw["dets"] = payload
        body = json.dumps(self.raw, indent=2, ensure_ascii=False)
        data = self.style.render(body)

        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self.path)

        # 記憶體值對齊檔案值,避免「存完再編輯」或「重開」出現落差
        for det, js in zip(self.dets, payload, strict=True):
            det.bbox = list(js["bbox"])
            det.ppe = js["ppe"]
        self._clean = self.dets_json()
        return True

    def snapshot(self) -> list[Det]:
        return [d.clone() for d in self.dets]

    def restore(self, snapshot: list[Det]) -> None:
        self.dets[:] = [d.clone() for d in snapshot]


def load_frame(path: Path) -> FrameLabel:
    data = path.read_bytes()
    bom = data.startswith(codecs.BOM_UTF8)
    text = data.decode("utf-8-sig" if bom else "utf-8")
    style = TextStyle(
        newline="\r\n" if "\r\n" in text else "\n",
        trailing_newline=text.endswith(("\n", "\r")),
        bom=bom,
    )
    raw = json.loads(text)
    dets = [
        Det(
            label=str(d.get("label", "person")),
            track_id=None if d.get("track_id") is None else int(d["track_id"]),
            ppe=d.get("ppe"),
            bbox=[float(v) for v in d.get("bbox", (0.0, 0.0, 0.0, 0.0))],
            raw=d,
        )
        for d in raw.get("dets", [])
    ]
    frame = FrameLabel(path=path, raw=raw, dets=dets, style=style)
    # 以「載入內容的 canonical 形式」當乾淨基準:沒編輯就不會被標成已改。
    frame._clean = frame.dets_json()
    return frame


@dataclass(slots=True)
class Interpolation:
    """一次補框的結果:要插入什麼、以及哪些洞因間距過大被略過。"""

    additions: list[tuple[int, Det]] = field(default_factory=list)
    skipped: list[tuple[int, int, int]] = field(default_factory=list)  # (seq0, seq1, 間距)

    @property
    def frame_indexes(self) -> list[int]:
        return sorted({i for i, _ in self.additions})


def interpolate_missing(
    frames: list[FrameLabel], label: str, track_id: int, max_gap: int
) -> Interpolation:
    """對同一條軌跡的相鄰錨點之間補上線性內插的框。

    軌跡的身分是 ``(label, track_id)`` 而不是單獨的 ``track_id``:下游
    ``eval_gt.py`` 讀 GT 時就把 person / drone 拆成兩個清單各自評估(MOT 還加
    id_prefix 區分),``gt_densify.py --drone-id`` 也只保證「同一架 drone 統一
    成一個號」,不保證跟 person 不撞。只認 track_id 的話,person#1 與 drone#1
    會被當成同一條軌跡——輕則洞被對方的框填掉而靜默不補,重則兩端錨點分屬不同
    label,內插出一個地面與天花板中點的捏造框。

    只在「間距 <= max_gap」時補。這個門檻同時擋兩件事:內插誤差(實測 20 幀間距
    IoU 中位 0.78、失準率 0.2%,30 幀就跳到 9.3%),以及**遮擋**——目標被擋住的
    區段人不會去標錨點,於是間距自然拉大而被拒絕,不會憑空生出錯誤的框。

    權重用 ``seq`` 差而非清單位置差,抽樣不連續的資料集也不會算歪。
    """
    anchors: list[tuple[int, int, Det]] = []
    for index, frame in enumerate(frames):
        match = next(
            (d for d in frame.dets if d.track_id == track_id and d.label == label),
            None,
        )
        if match is not None:
            anchors.append((index, frame.seq, match))

    result = Interpolation()
    for (i0, s0, d0), (i1, s1, d1) in zip(anchors, anchors[1:], strict=False):
        if i1 - i0 <= 1:
            continue
        span = s1 - s0
        if span > max_gap:
            result.skipped.append((s0, s1, span))
            continue
        for k in range(i0 + 1, i1):
            t = (frames[k].seq - s0) / max(span, 1)
            bbox = [d0.bbox[m] + (d1.bbox[m] - d0.bbox[m]) * t for m in range(4)]
            result.additions.append(
                (k, Det(label=d0.label, track_id=track_id, ppe=d0.ppe,
                        bbox=canonical_bbox(bbox)))
            )
    return result


@dataclass(slots=True)
class Remap:
    """一次 id 轉換的計畫:要改哪些框、哪些幀改完會同幀撞號。"""

    targets: list[tuple[int, Det]] = field(default_factory=list)  # (frame index, 要改號的框)
    conflicts: list[int] = field(default_factory=list)  # 改完會出現兩個同號框的 seq

    @property
    def frame_indexes(self) -> list[int]:
        return sorted({i for i, _ in self.targets})

    @property
    def box_count(self) -> int:
        return len(self.targets)


def plan_remap(frames: list[FrameLabel], label: str, old_id: int, new_id: int) -> Remap:
    """算出把 ``(label, old_id)`` 全域改成 ``new_id`` 會動到哪些框。

    tracker 斷軌時同一個目標會被切成兩個號碼,逐幀改號很慢,所以整條軌跡一次換號。

    軌跡身分是 ``(label, track_id)``,理由同 :func:`interpolate_missing`:只認
    track_id 的話,把 person#7 改成 #3 會連 drone#7 一起改號——而下游把 person /
    drone 拆成兩個清單各自評估,這種串號不會有人報錯,只會靜默算錯。

    ``conflicts`` 只收「同一幀**同時**有 ``(label, old_id)`` 與 ``(label, new_id)``」
    的 seq:改完那一幀會出現兩個同號框,通常代表這兩段其實不是同一個目標。反過來,
    只有 new_id 沒有 old_id 的幀不算衝突——斷軌合併的正常樣貌就是兩段各佔不同的幀。

    只算不改;套用交給呼叫方,才能在中間插入確認與快照。
    """
    result = Remap()
    for index, frame in enumerate(frames):
        hits = [d for d in frame.dets if d.label == label and d.track_id == old_id]
        if not hits:
            continue
        result.targets.extend((index, d) for d in hits)
        if any(d.label == label and d.track_id == new_id for d in frame.dets):
            result.conflicts.append(frame.seq)
    return result


def find_track(
    frames: list[FrameLabel], label: str | None, track_id: int
) -> list[tuple[int, int]]:
    """依幀序列出某條軌跡的每一次出現,回傳 ``(frame index, det index)``。

    ``label`` 給字串就限定該 label,``None`` 代表不分 label。預設應該指定:軌跡身分
    是 ``(label, track_id)``,理由同 :func:`interpolate_missing`——person#7 與
    drone#7 是兩條不同軌跡,混在一起追會在兩條線之間來回跳。留 ``None`` 是給
    「不確定號碼掛在哪個 label」時掃一遍用的。

    同一幀有多個同號框時**每個都列一筆**,不合併:那正是 ``has_duplicate_track``
    要人回頭看的狀況,搜尋逐個停下來才看得到第二個框在哪。

    每次呼叫重掃而不快取:dets 隨時在編輯(改號、補框、刪框),快取一定會跟不上,
    而掃一趟只是幾萬次欄位比對。
    """
    return [
        (i, k)
        for i, frame in enumerate(frames)
        for k, det in enumerate(frame.dets)
        if det.track_id == track_id and (label is None or det.label == label)
    ]


class UndoStack:
    """整份 dets 的快照式 undo/redo。

    快照法而非命令法:dets 每幀只有數十個小物件,複製成本可忽略,而拖曳縮放這類
    連續操作用命令物件很容易漏記反向狀態。每次「一個完整操作結束」才 commit 一筆。
    """

    def __init__(self, limit: int = 50) -> None:
        self.limit = max(2, limit)
        self._stack: list[list[Det]] = []
        self._idx = -1

    def reset(self, snapshot: list[Det]) -> None:
        self._stack = [[d.clone() for d in snapshot]]
        self._idx = 0

    def commit(self, snapshot: list[Det]) -> bool:
        if self._idx >= 0 and _same(self._stack[self._idx], snapshot):
            return False
        del self._stack[self._idx + 1 :]
        self._stack.append([d.clone() for d in snapshot])
        if len(self._stack) > self.limit:
            del self._stack[0]
        self._idx = len(self._stack) - 1
        return True

    @property
    def can_undo(self) -> bool:
        return self._idx > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._idx < len(self._stack) - 1

    def undo(self) -> list[Det] | None:
        if not self.can_undo:
            return None
        self._idx -= 1
        return [d.clone() for d in self._stack[self._idx]]

    def redo(self) -> list[Det] | None:
        if not self.can_redo:
            return None
        self._idx += 1
        return [d.clone() for d in self._stack[self._idx]]


def _same(a: list[Det], b: list[Det]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        x.label == y.label and x.track_id == y.track_id and x.ppe == y.ppe and x.bbox == y.bbox
        for x, y in zip(a, b, strict=True)
    )
