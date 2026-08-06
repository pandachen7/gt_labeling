"""驗收(GUI 層,offscreen):真的開資料夾、真的用滑鼠事件改框、真的存檔再開一次比對。

    uv run --project D:\\ws\\gt_labeling python scripts/verify_gui.py <gt_sample_root>
"""

from __future__ import annotations

import collections
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from gt_labeling.canvas import COLOR_DRONE, det_color
from gt_labeling.model import (
    Det,
    canonical_bbox,
    interpolate_missing,
    load_frame,
    plan_remap,
)
from gt_labeling.window import MainWindow

FAILURES: list[str] = []
OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        FAILURES.append(message)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def prepare_workdir(source: Path, tmp: Path) -> Path:
    work = tmp / "gt_sample"
    work.mkdir(parents=True)
    shutil.copytree(source / "labels", work / "labels")
    shutil.copytree(source / "frames", work / "frames")
    return work


def find_empty_spot(canvas, span: int = 60) -> QPoint | None:
    """在影像範圍內找一個既不壓到框也不壓到控制點的起點,且往右下 span px 仍在影像內。"""
    from PyQt6.QtCore import QPointF

    image_rect = canvas.tf.image_rect()
    for ratio_y in (0.15, 0.3, 0.05, 0.5, 0.7):
        for ratio_x in (0.05, 0.15, 0.3, 0.45, 0.6):
            x = image_rect.left() + image_rect.width() * ratio_x
            y = image_rect.top() + image_rect.height() * ratio_y
            if not image_rect.contains(x + span, y + span):
                continue
            corners = [QPointF(x, y), QPointF(x + span, y + span),
                       QPointF(x + span, y), QPointF(x, y + span)]
            if any(canvas._hit_box(p) is not None or canvas._hit_handle(p) is not None
                   for p in corners):
                continue
            return QPoint(int(x), int(y))
    return None


def drag(canvas, start: QPoint, end: QPoint, steps: int = 6) -> None:
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for i in range(1, steps + 1):
        QTest.mouseMove(
            canvas,
            QPoint(
                start.x() + round((end.x() - start.x()) * i / steps),
                start.y() + round((end.y() - start.y()) * i / steps),
            ),
        )
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\ws\detect_stream\out\gt_sample")
    if not (source / "labels").is_dir():
        print(f"找不到 {source}\\labels")
        return 2

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="gt_gui_"))
    try:
        # QSettings 導到暫存目錄,不污染使用者真正的設定。
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp / "cfg"))

        work = prepare_workdir(source, tmp)
        app = QApplication(sys.argv[:1])
        app.setOrganizationName("asys")
        app.setApplicationName("gt_labeling_verify")

        window = MainWindow(cache_size=8)
        window.resize(1600, 900)
        window.set_band(0.50, 0.90)
        window.show()
        app.processEvents()

        section("開資料夾")
        check(window.open_root(work), "open_root 成功")
        check(len(window.frames) == 75, f"載入 75 幀(實際 {len(window.frames)})")
        canvas = window.canvas
        app.processEvents()
        check(canvas.width() > 400 and canvas.height() > 300,
              f"canvas 有實際尺寸 {canvas.width()}x{canvas.height()}")
        check(canvas.frame is not None and canvas.frame.size == (3840, 1920),
              "首幀尺寸 3840x1920")

        section("逐幀瀏覽 75 幀")
        t0 = time.perf_counter()
        visited = 0
        for _ in range(74):
            window._navigate(1)
            app.processEvents()
            visited += 1
        first_pass = time.perf_counter() - t0
        check(window.index == 74, f"走到最後一幀(index={window.index})")
        check(visited == 74, "74 次前進全部成功")
        print(f"       首輪(含 JPEG 解碼)共 {first_pass * 1000:.0f} ms,"
              f"平均 {first_pass / 74 * 1000:.1f} ms/幀")

        # 快取命中路徑:在小範圍來回切,不應再解碼。
        window._goto(40)
        app.processEvents()
        t0 = time.perf_counter()
        for i in range(20):
            window._goto(40 + (i % 3))
            app.processEvents()
        cached = (time.perf_counter() - t0) / 20
        print(f"       快取命中切幀平均 {cached * 1000:.1f} ms/幀")
        check(cached < 0.030, f"快取命中切幀 < 30ms(實際 {cached * 1000:.1f} ms)")

        t0 = time.perf_counter()
        for _ in range(30):
            canvas.grab()
        repaint = (time.perf_counter() - t0) / 30
        print(f"       重繪平均 {repaint * 1000:.1f} ms(不重新解碼 JPEG)")
        check(repaint < 0.050, f"重繪 < 50ms(實際 {repaint * 1000:.1f} ms)")

        section("Home / End 跳首末幀")
        window._goto(len(window.frames) // 2)
        app.processEvents()
        check(window.act_first.isEnabled() and window.act_last.isEnabled(),
              "在中間幀時「首幀」「末幀」都可用")
        # 真的按鍵(不是只 trigger action):順便驗快捷鍵確實接在畫布上。
        canvas.setFocus()
        app.processEvents()
        QTest.keyClick(canvas, Qt.Key.Key_Home)
        app.processEvents()
        check(window.index == 0, f"按 Home 跳到第一幀(實際 index={window.index})")
        check(not window.act_first.isEnabled(), "已在第一幀時「首幀」停用")
        QTest.keyClick(canvas, Qt.Key.Key_End)
        app.processEvents()
        check(window.index == len(window.frames) - 1,
              f"按 End 跳到最後一幀(實際 index={window.index})")
        check(not window.act_last.isEnabled(), "已在最後一幀時「末幀」停用")
        # 綁在畫布而非視窗,jump_edit 打字時的 Home / End 才不會被搶走。
        check(window.act_first.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
              and window.act_first in canvas.actions(),
              "Home / End 綁在畫布(WidgetShortcut),不攔 jump_edit 的行首 / 行尾")

        section("滑鼠拖曳:移動框")
        target_index = next(i for i, f in enumerate(window.frames) if len(f.dets) >= 5)
        window._goto(target_index)
        app.processEvents()
        frame = window.frames[target_index]

        canvas.select(0)
        det = frame.dets[0]
        before_rect = canvas.tf.n2v_rect(det.bbox)
        before_bbox = list(det.bbox)
        start = before_rect.center().toPoint()
        end = QPoint(start.x() + 24, start.y() + 16)
        drag(canvas, start, end)
        app.processEvents()

        after_rect = canvas.tf.n2v_rect(det.bbox)
        dx_px = after_rect.center().x() - before_rect.center().x()
        dy_px = after_rect.center().y() - before_rect.center().y()
        check(abs(dx_px - 24) < 1.5 and abs(dy_px - 16) < 1.5,
              f"框中心移動 ({dx_px:.2f}, {dy_px:.2f}) px,目標 (24, 16)")
        check(abs(after_rect.width() - before_rect.width()) < 0.6
              and abs(after_rect.height() - before_rect.height()) < 0.6,
              "移動不改變框尺寸(無變形)")
        check(det.bbox != before_bbox and all(0.0 <= v <= 1.0 for v in det.bbox),
              f"bbox 已更新且仍歸一化 {det.bbox}")
        check(frame.dirty, "移動後標記未存")

        section("滑鼠拖曳:控制點縮放")
        det = frame.dets[0]
        rect = canvas.tf.n2v_rect(det.bbox)
        br = rect.bottomRight().toPoint()
        left_before, top_before = rect.left(), rect.top()
        drag(canvas, br, QPoint(br.x() + 30, br.y() + 20))
        app.processEvents()
        rect2 = canvas.tf.n2v_rect(det.bbox)
        check(abs(rect2.right() - rect.right() - 30) < 1.5
              and abs(rect2.bottom() - rect.bottom() - 20) < 1.5,
              f"右下邊移動 ({rect2.right() - rect.right():.2f}, "
              f"{rect2.bottom() - rect.bottom():.2f}) px,目標 (30, 20)")
        check(abs(rect2.left() - left_before) < 0.6 and abs(rect2.top() - top_before) < 0.6,
              "左上邊沒有跟著跑")

        section("新框預設選擇器:drone")
        count_before = len(frame.dets)
        expected_id = max(
            (d.track_id for d in frame.dets if d.track_id is not None), default=-1
        ) + 1
        window.new_panel.set_label("drone")
        app.processEvents()
        check(window.new_panel.defaults() == ("drone", None),
              f"選擇器回報 drone 預設 {window.new_panel.defaults()}")
        empty = find_empty_spot(canvas)
        check(empty is not None, f"找到影像內的空白起點 {empty}")
        drag(canvas, empty, QPoint(empty.x() + 60, empty.y() + 40))
        app.processEvents()
        check(len(frame.dets) == count_before + 1, f"框數 {count_before} -> {len(frame.dets)}")
        new_det = frame.dets[-1]
        check(new_det.label == "drone" and new_det.ppe is None,
              f"新框是 drone / ppe=null(實際 {new_det.label} / {new_det.ppe})")
        check(new_det.track_id == expected_id,
              f"track_id = 本幀最大 +1 = {expected_id}(實際 {new_det.track_id})")
        check(det_color(new_det) == COLOR_DRONE, "drone 框畫成紅色")
        check(not new_det.pending, "drone 新框不再是待補")
        check(canvas.selected_index == len(frame.dets) - 1, "新框自動被選取")

        section("新框預設選擇器:person")
        count_before = len(frame.dets)
        window.new_panel.set_label("person")
        app.processEvents()
        check(window.new_panel.defaults() == ("person", "ng"),
              f"選擇器回報 person 預設 {window.new_panel.defaults()}")
        empty = find_empty_spot(canvas)
        check(empty is not None, f"找到第二個空白起點 {empty}")
        drag(canvas, empty, QPoint(empty.x() + 60, empty.y() + 40))
        app.processEvents()
        check(len(frame.dets) == count_before + 1, f"框數 {count_before} -> {len(frame.dets)}")
        new_det = frame.dets[-1]
        check(new_det.label == "person" and new_det.ppe == "ng",
              f"新框是 person / ppe=ng(實際 {new_det.label} / {new_det.ppe})")
        check(new_det.track_id == expected_id + 1,
              f"第二個新框續號 {expected_id + 1}(實際 {new_det.track_id})")
        check(not new_det.pending, "person 新框不再是待補")

        section("新框預設:沿用選取框的 track_id")
        check(window.new_panel.follow_id() is None, "沒選「沿用」時 follow_id() 回 None")

        anchor_pos = next(k for k, d in enumerate(frame.dets) if d.track_id is not None)
        canvas.select(anchor_pos)
        app.processEvents()
        anchor_id = frame.dets[anchor_pos].track_id
        check(window.new_panel.follow_radio.isEnabled(),
              f"點過框後「沿用」可選,候選 = #{anchor_id}")
        check(f"#{anchor_id}" in window.new_panel.follow_radio.text(),
              f"radio 顯示候選號碼:{window.new_panel.follow_radio.text()!r}")

        # 兩組 radio 必須互不干擾:切 label 不能把 track_id 的選擇彈掉。
        window.new_panel.follow_radio.setChecked(True)
        window.new_panel.set_label("drone")
        app.processEvents()
        check(window.new_panel.follow_radio.isChecked() and window.new_panel.label() == "drone",
              "切換 label 不會把「沿用」彈掉(QButtonGroup 分組正確)")
        check(window.new_panel.follow_id() == anchor_id,
              f"follow_id() = {window.new_panel.follow_id()}(預期 {anchor_id})")

        count_before = len(frame.dets)
        empty = find_empty_spot(canvas)
        check(empty is not None, f"找到第三個空白起點 {empty}")
        drag(canvas, empty, QPoint(empty.x() + 60, empty.y() + 40))
        app.processEvents()
        check(len(frame.dets) == count_before + 1, f"框數 {count_before} -> {len(frame.dets)}")
        check(frame.dets[-1].track_id == anchor_id,
              f"新框沿用 #{anchor_id}(實際 {frame.dets[-1].track_id}),而非本幀最大 +1")

        window.new_panel.auto_radio.setChecked(True)
        window.new_panel.set_label("person")
        app.processEvents()
        check(window.new_panel.follow_id() is None, "切回自動後不再沿用")
        canvas.delete_selected()
        app.processEvents()

        section("面板編輯 + Delete")
        window.det_panel.track_edit.setText("777")
        window.det_panel.track_edit.editingFinished.emit()
        app.processEvents()
        check(frame.dets[-1].track_id == 777, f"track_id 寫入 777(實際 {frame.dets[-1].track_id})")

        window.det_panel.label_box.setCurrentText("drone")
        app.processEvents()
        check(frame.dets[-1].label == "drone", "label 改成 drone")
        check(not window.det_panel.ppe_box.isEnabled(), "drone 的 ppe 欄位被鎖住")

        window.det_panel.label_box.setCurrentText("person")
        window.det_panel.ppe_box.setCurrentIndex(window.det_panel.ppe_box.findData("ng"))
        app.processEvents()
        check(frame.dets[-1].ppe == "ng", f"ppe 寫入 ng(實際 {frame.dets[-1].ppe})")

        count_before = len(frame.dets)
        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        app.processEvents()
        check(len(frame.dets) == count_before - 1, f"Delete 刪掉一框 -> {len(frame.dets)}")

        section("復原/重做 25 步")
        baseline = frame.dets_json()
        canvas.select(0)
        for i in range(25):
            frame.dets[0].bbox = [
                min(max(v + 0.0007 * (i + 1), 0.0), 1.0) for v in frame.dets[0].bbox
            ]
            canvas.detsEdited.emit()
        app.processEvents()
        after_edits = frame.dets_json()
        check(after_edits != baseline, "25 次編輯確實改變了狀態")

        for _ in range(25):
            window._undo()
        app.processEvents()
        check(frame.dets_json() == baseline, "連續 25 次復原回到原狀")

        for _ in range(25):
            window._redo()
        app.processEvents()
        check(frame.dets_json() == after_edits, "連續 25 次重做回到編輯後狀態")

        section("存檔 -> 從磁碟重開 -> 逐值比對")
        in_memory = frame.dets_json()
        label_path = frame.path
        window._save_current()
        app.processEvents()
        check(not frame.dirty, "存檔後不再是未存")

        reopened = load_frame(label_path)
        check(reopened.dets_json() == in_memory, "重開後 dets 與存檔前逐值相同")
        check(all(0.0 <= v <= 1.0 for d in reopened.dets_json() for v in d["bbox"]),
              "重開後 bbox 仍是歸一化值")

        # 重開整個資料集,確認畫面上的框位置(view rect)完全一致。
        rects_before = [canvas.tf.n2v_rect(d.bbox) for d in frame.dets]
        zoom_before, off_before = canvas.tf.zoom, (canvas.tf.off_x, canvas.tf.off_y)
        check(window.open_root(work), "重新開啟同一個資料夾")
        window._goto(target_index)
        app.processEvents()
        canvas.tf.zoom = zoom_before
        canvas.tf.off_x, canvas.tf.off_y = off_before
        reloaded_frame = window.frames[target_index]
        rects_after = [canvas.tf.n2v_rect(d.bbox) for d in reloaded_frame.dets]
        same = len(rects_before) == len(rects_after) and all(
            abs(a.left() - b.left()) < 1e-9 and abs(a.top() - b.top()) < 1e-9
            and abs(a.right() - b.right()) < 1e-9 and abs(a.bottom() - b.bottom()) < 1e-9
            for a, b in zip(rects_before, rects_after)
        )
        check(same, "重開後每個框的畫面位置完全一致(零漂移)")

        section("內插補框")
        # 挖一個真的洞:找一個橫跨多幀的 track,把中間幾幀的框拿掉,再補回來比對。
        counts = collections.Counter(
            # 軌跡身分是 (label, track_id),不是單獨的 track_id
            (d.label, d.track_id)
            for f in window.frames for d in f.dets if d.track_id is not None
        )
        probe = next((k for k, c in counts.most_common() if c >= 6), None)
        check(probe is not None, f"找到橫跨多幀的 track{probe}")
        probe_label, probe_tid = probe

        def on_track(det) -> bool:
            return det.track_id == probe_tid and det.label == probe_label

        rows = [i for i, f in enumerate(window.frames) if any(on_track(d) for d in f.dets)]
        hole = rows[1:4]
        truth, origin = {}, {}
        for i in hole:
            frame_i = window.frames[i]
            pos = next(k for k, d in enumerate(frame_i.dets) if on_track(d))
            truth[i] = list(frame_i.dets[pos].bbox)
            origin[i] = (pos, frame_i.dets[pos])
            del frame_i.dets[pos]
        check(all(not any(on_track(d) for d in window.frames[i].dets) for i in hole),
              f"挖掉 {len(hole)} 幀的 {probe_label}#{probe_tid}")

        window.interp_panel.set_state(True, 2)
        window._goto(rows[0])
        window.canvas.select(
            next(k for k, d in enumerate(window.frames[rows[0]].dets) if on_track(d))
        )
        app.processEvents()
        check(window.act_interp.isEnabled(), "有資料時「補框」動作可觸發(不靜默失敗)")
        gap_seq = window.frames[rows[4]].seq - window.frames[rows[0]].seq
        tight = interpolate_missing(window.frames, probe_label, probe_tid, 2)
        check(not tight.additions and tight.skipped,
              f"門檻 2 幀時整個洞被跳過(實際 seq 間距 {gap_seq},skipped={len(tight.skipped)})")

        window.interp_panel.set_state(True, 100000)
        app.processEvents()
        loose = interpolate_missing(window.frames, probe_label, probe_tid, 100000)
        check(len(loose.additions) >= len(hole),
              f"放寬門檻後算出 {len(loose.additions)} 個要補的框(挖掉的有 {len(hole)} 個)")
        window.apply_interpolation(loose)
        app.processEvents()
        filled = {}
        for i in hole:
            got = next((d for d in window.frames[i].dets if d.track_id == probe_tid), None)
            if got is not None:
                filled[i] = got
        check(len(filled) == len(hole), f"套用後補回 {len(filled)}/{len(hole)} 幀")

        if len(filled) == len(hole):
            # 驗的是內插的數學正確性,不是與真實框的 IoU:gt_sample 是 5 秒抽樣,
            # 人走 5 秒後框本來就不重疊,那是資料特性,不是程式對錯。
            a_i, b_i = rows[0], rows[4]
            a = next(d for d in window.frames[a_i].dets if d.track_id == probe_tid)
            b = next(d for d in window.frames[b_i].dets if d.track_id == probe_tid)
            sa, sb = window.frames[a_i].seq, window.frames[b_i].seq
            exact = True
            for i in hole:
                t = (window.frames[i].seq - sa) / (sb - sa)
                want = canonical_bbox(
                    [a.bbox[m] + (b.bbox[m] - a.bbox[m]) * t for m in range(4)]
                )
                exact = exact and filled[i].bbox == want
            check(exact, f"補回的 bbox 等於兩錨點(seq {sa}/{sb})的線性內插值")

            src = next(d for d in window.frames[rows[0]].dets if d.track_id == probe_tid)
            check(all(f.label == src.label and f.ppe == src.ppe for f in filled.values()),
                  f"補回的框沿用錨點屬性({src.label} / {src.ppe})")
            ious = [box_iou(filled[i].bbox, truth[i]) for i in hole]
            print(f"       參考:與真實框 IoU 中位 {sorted(ious)[len(ious) // 2]:.3f}"
                  f"(5 秒抽樣資料本來就低,逐幀資料才有意義)")

        window._undo_last_bulk()
        app.processEvents()
        check(all(not any(d.track_id == probe_tid for d in window.frames[i].dets)
                  for i in hole), "Ctrl+Shift+I 整組復原,補的框全部消失")

        # 放回原位(不是 append),否則 dets 順序改變會讓該幀永遠是未存狀態。
        for i in hole:
            pos, det_i = origin[i]
            window.frames[i].dets.insert(pos, det_i)
        check(not any(window.frames[i].dirty for i in hole), "還原後這幾幀回到已存狀態")

        window.interp_panel.set_state(False, 20)
        app.processEvents()
        before_counts = [len(f.dets) for f in window.frames]
        window._interpolate()  # 未啟用時應直接說明並返回,不彈對話框、不動資料
        app.processEvents()
        check([len(f.dets) for f in window.frames] == before_counts,
              "取消勾選後按補框不會改動任何資料")
        check("勾選" in window.statusBar().currentMessage(),
              f"且有說明原因:{window.statusBar().currentMessage()!r}")

        section("改 id:把斷軌的兩段接回同一個號碼")
        # 前置不足時要說明原因並原地返回,而不是靜默失敗、也不是彈對話框把 offscreen 卡死。
        canvas.select(-1)
        app.processEvents()
        ids_before = [(d.label, d.track_id) for f in window.frames for d in f.dets]
        window._remap_track()
        app.processEvents()
        check([(d.label, d.track_id) for f in window.frames for d in f.dets] == ids_before,
              "沒選框時按「改 id」不動任何資料")
        check("先點選" in window.statusBar().currentMessage(),
              f"且有說明原因:{window.statusBar().currentMessage()!r}")
        check(window.act_remap.isEnabled(), "有資料時「改 id」動作可觸發(不靜默失敗)")

        # 造一條斷軌:把 probe track 的後半段改成一個全資料集沒人用的號碼。
        spare = max(d.track_id for f in window.frames for d in f.dets
                    if d.track_id is not None) + 1
        tail = {i: [pos for pos, d in enumerate(window.frames[i].dets) if on_track(d)]
                for i in rows[len(rows) // 2:]}
        for i, positions in tail.items():
            for pos in positions:
                window.frames[i].dets[pos].track_id = spare
        broken = sum(len(p) for p in tail.values())
        check(broken >= 1, f"把後 {len(tail)} 幀的 {broken} 個框改成 #{spare},製造斷軌")

        # 同號但不同 label 的框:認軌是 (label, track_id),它絕對不能被一起改掉。
        decoy_row = next(iter(tail))
        decoy = Det(label="drone", track_id=spare, ppe=None,
                    bbox=[0.90, 0.10, 0.95, 0.15])
        window.frames[decoy_row].dets.append(decoy)

        plan = plan_remap(window.frames, probe_label, spare, probe_tid)
        check(plan.frame_indexes == sorted(tail),
              f"算出要改 {len(tail)} 幀(實際 {len(plan.frame_indexes)})")
        check(plan.box_count == broken,
              f"算出要改 {broken} 個框(實際 {plan.box_count})")
        check(not plan.conflicts,
              f"斷軌兩段不重疊 → 沒有撞號警告(實際 {plan.conflicts})")

        window.apply_remap(plan, probe_label, spare, probe_tid)
        app.processEvents()
        check(all(sum(1 for d in window.frames[i].dets
                      if d.label == probe_label and d.track_id == probe_tid)
                  == len(positions) for i, positions in tail.items()),
              f"後半段全部接回 {probe_label} #{probe_tid}")
        check(not any(d.label == probe_label and d.track_id == spare
                      for f in window.frames for d in f.dets),
              f"整份資料集不再有 {probe_label} #{spare}")
        check(decoy.track_id == spare,
              f"同號的 drone #{spare} 沒被波及(認軌是 (label, track_id) 而非 track_id)")
        check(window.frames[decoy_row].dirty, "改號後該幀標記未存")

        # 撞號:同一幀同時有來源號與目標號時必須警告 —— 那多半代表兩段不是同一個目標。
        overlap_row = rows[0]
        extra = Det(label=probe_label, track_id=spare, ppe="ng",
                    bbox=[0.05, 0.05, 0.09, 0.12])
        window.frames[overlap_row].dets.append(extra)
        clash = plan_remap(window.frames, probe_label, spare, probe_tid)
        check(clash.conflicts == [window.frames[overlap_row].seq],
              f"同幀已有 #{probe_tid} 時列入撞號警告:seq {clash.conflicts}")
        check(window.frames[rows[1]].seq not in clash.conflicts,
              "只有目標號、沒有來源號的幀不算撞號(斷軌另一段本來就該保留)")
        window.frames[overlap_row].dets.remove(extra)

        window._undo_last_bulk()
        app.processEvents()
        check(all(all(window.frames[i].dets[pos].track_id == spare for pos in positions)
                  for i, positions in tail.items()),
              f"Ctrl+Shift+I 整組復原,後半段回到斷軌狀態 #{spare}")

        # 還原成磁碟上的樣子:號碼改回去、拿掉造出來的 drone。
        for i, positions in tail.items():
            for pos in positions:
                window.frames[i].dets[pos].track_id = probe_tid
        window.frames[decoy_row].dets[:] = [
            d for d in window.frames[decoy_row].dets
            if not (d.label == "drone" and d.track_id == spare)
        ]
        check(not any(window.frames[i].dirty for i in tail),
              "還原後這幾幀回到已存狀態")

        section("清單待補標記")
        # 樣本資料不保證含 null,自己在記憶體造一個待補狀態再還原,驗證與資料內容無關。
        probe_row = next(
            (i for i, f in enumerate(window.frames)
             if f.dets and not f.has_null_track and not f.has_null_ppe),
            None,
        )
        check(probe_row is not None, "找到一幀沒有待補的列當基準")
        if probe_row is not None:
            probe_frame = window.frames[probe_row]
            check("ID" not in window.list_panel.list.item(probe_row).text(),
                  "沒有待補的列不帶 ID 標記")

            original_id = probe_frame.dets[0].track_id
            probe_frame.dets[0].track_id = None
            window.list_panel.refresh_row(probe_row, probe_frame)
            check(probe_frame.has_null_track, "把 track_id 設成 null 後該幀標記為待補")
            check("ID" in window.list_panel.list.item(probe_row).text(),
                  f"該列文字含 ID 標記:{window.list_panel.list.item(probe_row).text()!r}")

            probe_frame.dets[0].track_id = original_id
            window.list_panel.refresh_row(probe_row, probe_frame)
            check(not probe_frame.dirty and "ID" not in
                  window.list_panel.list.item(probe_row).text(),
                  "還原 track_id 後標記消失且不算未存")
        window.list_panel.refresh_summary(window.frames)
        print(f"       摘要:{window.list_panel.summary.text().replace(chr(10), ' | ')}")

        section("輸出截圖")
        OUT_DIR.mkdir(exist_ok=True)
        window._goto(target_index)
        canvas.fit_view()
        app.processEvents()
        shot = OUT_DIR / "verify_canvas.png"
        check(canvas.grab().save(str(shot)), f"canvas 截圖 -> {shot}")
        window_shot = OUT_DIR / "verify_window.png"
        check(window.grab().save(str(window_shot)), f"視窗截圖 -> {window_shot}")

        # 關 autosave 後若還有未存幀,closeEvent 會彈確認框把 offscreen 測試卡死。
        for i, f in enumerate(window.frames):
            if f.dirty:
                window._reload_frame(i)
        window.act_autosave.setChecked(False)
        window.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

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
