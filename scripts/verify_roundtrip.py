"""驗收:存檔往返不漂移、非 dets 欄位一字未動、座標換算可逆。

    uv run --project D:\\ws\\gt_labeling python scripts/verify_roundtrip.py <gt_sample_root>
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QPointF, QSize

from gt_labeling.dataset import scan_root
from gt_labeling.model import Det, canonical_bbox, load_frame
from gt_labeling.transform import ViewTransform

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def non_dets(raw: dict) -> list[tuple[str, object]]:
    """含 key 順序的非 dets 欄位快照。"""
    return [(k, v) for k, v in raw.items() if k != "dets"]


def test_unchanged_save_is_byte_identical(root: Path) -> None:
    section("未編輯存檔:byte 完全相同")
    entries = scan_root(root)
    check(len(entries) == 75, f"掃到 75 幀(實際 {len(entries)})")

    wrote = 0
    identical = 0
    for entry in entries:
        before = entry.label_path.read_bytes()
        frame = load_frame(entry.label_path)
        if frame.save():
            wrote += 1
        if entry.label_path.read_bytes() == before:
            identical += 1
    check(wrote == 0, f"沒有改動就不寫檔(實際寫了 {wrote} 個)")
    check(identical == len(entries), f"{identical}/{len(entries)} 檔 byte 相同")

    forced = 0
    for entry in entries:
        before = entry.label_path.read_bytes()
        frame = load_frame(entry.label_path)
        frame.save(force=True)
        if entry.label_path.read_bytes() == before:
            forced += 1
    check(forced == len(entries), f"強制重寫後仍 byte 相同:{forced}/{len(entries)}")


def test_edit_roundtrip(root: Path) -> None:
    section("編輯 -> 存檔 -> 重開:座標零漂移、非 dets 欄位不動")
    entries = scan_root(root)
    target = next(e for e in entries if len(load_frame(e.label_path).dets) >= 5)
    original_bytes = target.label_path.read_bytes()
    original_raw = json.loads(original_bytes.decode("utf-8"))

    frame = load_frame(target.label_path)
    before_fields = non_dets(frame.raw)

    # 模擬各種編輯:移動、縮放、改欄位、新增、刪除。
    frame.dets[0].bbox = [v + 0.013571 for v in frame.dets[0].bbox]
    frame.dets[1].bbox[2] += 0.024999
    frame.dets[2].track_id = None
    frame.dets[3].ppe = None
    frame.dets[4].label = "drone"
    frame.dets.append(Det(label="person", track_id=None, ppe=None,
                          bbox=[0.111116, 0.222224, 0.333338, 0.444446]))
    removed = frame.dets.pop(0)
    check(removed is not None, "刪除一個框")

    in_memory = frame.dets_json()
    check(frame.dirty, "編輯後被標記為未存")
    check(frame.save(), "存檔實際寫入")
    check(not frame.dirty, "存檔後不再是未存狀態")

    # 存檔後記憶體值必須等於寫出去的值,否則「存完繼續編輯」會偷偷漂移。
    check(frame.dets_json() == in_memory, "記憶體值已對齊寫出的值")

    reopened = load_frame(target.label_path)
    check(reopened.dets_json() == in_memory, "重開後每個 bbox 與存檔前完全相同")
    check(non_dets(reopened.raw) == before_fields, "非 dets 欄位(含 key 順序)一字未動")
    check(
        non_dets(reopened.raw) == non_dets(original_raw),
        "非 dets 欄位與原始檔相同",
    )

    saved_raw = json.loads(target.label_path.read_bytes().decode("utf-8"))
    all_normalized = all(
        0.0 <= v <= 1.0 for d in saved_raw["dets"] for v in d["bbox"]
    )
    check(all_normalized, "存出的 bbox 全在 [0,1](仍是歸一化,不是像素)")
    five_dp = all(
        round(v, 5) == v for d in saved_raw["dets"] for v in d["bbox"]
    )
    check(five_dp, "存出的 bbox 都是 5 位小數")
    ordered = all(
        d["bbox"][0] < d["bbox"][2] and d["bbox"][1] < d["bbox"][3] for d in saved_raw["dets"]
    )
    check(ordered, "存出的 bbox 都滿足 x1<x2 且 y1<y2")
    check(
        all(d["ppe"] is None for d in saved_raw["dets"] if d["label"] != "person"),
        "非 person 的 ppe 一律 null",
    )

    # 位元組層:只有 dets 區段變了 —— 用原始檔換上新 dets 重新序列化應完全吻合。
    rebuilt = dict(original_raw)
    rebuilt["dets"] = saved_raw["dets"]
    expected = json.dumps(rebuilt, indent=2, ensure_ascii=False).replace("\n", "\r\n")
    check(
        expected.encode("utf-8") == target.label_path.read_bytes(),
        "檔案位元組 = 原始檔僅替換 dets(其餘連行尾都沒動)",
    )

    # 反覆存/開多輪不得再有任何變化。
    stable = True
    snapshot = target.label_path.read_bytes()
    for _ in range(5):
        cycle = load_frame(target.label_path)
        cycle.save(force=True)
        if target.label_path.read_bytes() != snapshot:
            stable = False
            break
    check(stable, "連續 5 輪存/開位元組不再變動")


def test_canonical_bbox() -> None:
    section("canonical_bbox 邊界處理")
    check(canonical_bbox([0.6, 0.7, 0.2, 0.3]) == [0.2, 0.3, 0.6, 0.7], "反向座標會被排序")
    check(canonical_bbox([-0.5, -0.2, 1.8, 2.0]) == [0.0, 0.0, 1.0, 1.0], "超界會被 clamp")
    check(canonical_bbox([0.123456789, 0.5, 0.987654321, 0.6])
          == [0.12346, 0.5, 0.98765, 0.6], "round 到 5 位")
    degenerate = canonical_bbox([0.5, 0.5, 0.500001, 0.500001])
    check(degenerate[0] < degenerate[2] and degenerate[1] < degenerate[3],
          f"退化框仍保證非零寬高 {degenerate}")
    at_edge = canonical_bbox([1.0, 1.0, 1.0, 1.0])
    check(at_edge[0] < at_edge[2] and at_edge[1] < at_edge[3],
          f"貼右下角的退化框也能修好 {at_edge}")
    check(canonical_bbox(canonical_bbox([0.123456789, 0.5, 0.987654321, 0.6]))
          == canonical_bbox([0.123456789, 0.5, 0.987654321, 0.6]), "canonical 是冪等的")


def test_transform_roundtrip() -> None:
    section("ViewTransform 可逆性(座標漂移防線)")
    tf = ViewTransform()
    tf.set_image_size(3840, 1920)
    view = QSize(1600, 900)
    tf.fit(view)

    worst = 0.0
    for zoom_step in range(-12, 13):
        tf.zoom_by(1.12**zoom_step, QPointF(731.0, 409.0), tf.fit_zoom(view) * 0.25)
        tf.pan_by(-37.0 * zoom_step, 21.0 * zoom_step)
        tf.clamp_offset(view)
        for nx in (0.0, 0.00013, 0.25, 0.5, 0.749997, 1.0):
            for ny in (0.0, 0.31, 0.68194, 1.0):
                back = tf.v2n_point(tf.n2v(nx, ny))
                worst = max(worst, abs(back.x() - nx), abs(back.y() - ny))
    check(worst < 1e-12, f"v2n(n2v(n)) 最大誤差 {worst:.3e} < 1e-12")

    bbox = [0.54063, 0.58415, 0.57641, 0.67782]
    worst_bbox = 0.0
    for zoom in (0.05, 0.1875, 0.41667, 1.0, 3.0, 12.0):
        tf.zoom = zoom
        tf.off_x, tf.off_y = -1234.5, 678.25
        back = tf.v2n_rect(tf.n2v_rect(bbox))
        worst_bbox = max(worst_bbox, *(abs(a - b) for a, b in zip(back, bbox)))
    check(worst_bbox < 1e-12, f"bbox 往返最大誤差 {worst_bbox:.3e} < 1e-12")

    # 拖曳是「起點 bbox + normalized 位移」重算,連續 200 次微動不得累積誤差。
    tf.zoom = 0.41667
    tf.off_x, tf.off_y = 12.0, 34.0
    origin = list(bbox)
    start = tf.n2v(origin[0], origin[1])
    for i in range(1, 201):
        now = QPointF(start.x() + i * 0.37, start.y() + i * 0.19)
        n0 = tf.v2n_point(start)
        n1 = tf.v2n_point(now)
        dx, dy = n1.x() - n0.x(), n1.y() - n0.y()
        moved = [origin[0] + dx, origin[1] + dy, origin[2] + dx, origin[3] + dy]
    width_drift = abs((moved[2] - moved[0]) - (origin[2] - origin[0]))
    height_drift = abs((moved[3] - moved[1]) - (origin[3] - origin[1]))
    check(width_drift < 1e-15 and height_drift < 1e-15,
          f"200 次拖曳後框尺寸漂移 {width_drift:.3e} / {height_drift:.3e}")


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\ws\detect_stream\out\gt_sample")
    if not (source / "labels").is_dir():
        print(f"找不到 {source}\\labels")
        return 2

    with tempfile.TemporaryDirectory(prefix="gt_verify_") as tmp:
        work = Path(tmp) / "gt_sample"
        # 只複製 labels(影像不需要),frames 建空目錄讓 scan_root 通過。
        (work / "frames").mkdir(parents=True)
        shutil.copytree(source / "labels", work / "labels")
        print(f"工作副本:{work}")

        test_unchanged_save_is_byte_identical(work)
        test_edit_roundtrip(work)

    test_canonical_bbox()
    test_transform_roundtrip()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失敗 {len(FAILURES)} 項:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("全部通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
