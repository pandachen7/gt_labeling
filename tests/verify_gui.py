"""驗收(GUI 層,offscreen):真的開資料夾、真的用滑鼠事件改框、真的存檔再開一次比對。

    uv run --project D:\\ws\\gt_labeling python tests/verify_gui.py <gt_root>

<gt_root> 是含 ``frames/`` 與 ``labels/`` 的單一資料夾。預設指向 per-frame GT 的
第一個時段;那份 GT 按 ``000-020s`` 這樣分段,每段自成一個 root,要驗別段就把
路徑換掉。原始資料一律只讀:所有編輯都發生在 ``prepare_workdir`` 複製出來的
暫存副本上。
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QPoint, QSettings, Qt, QTimer
from PyQt6.QtGui import QKeySequence
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QDialogButtonBox, QMessageBox

from gt_labeling.canvas import COLOR_DRONE, det_color
from gt_labeling.model import (
    Det,
    find_track,
    interpolate_missing,
    load_frame,
    plan_remap,
    plan_track_delete,
)
from gt_labeling.panel import NewDetPanel
from gt_labeling.window import MainWindow

FAILURES: list[str] = []
OUT_DIR = Path(__file__).resolve().parents[1] / "out"
# 逐幀瀏覽最多走幾幀(夠測解碼成本就好,不必走完整份資料集)。
WALK_FRAMES = 74
# 用 Shift+↓ 拉選取範圍時最多拉幾列(同理:夠證明拉得出連續範圍就好)。
SHIFT_RANGE_ROWS = 40


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


def catch_modal(store: list[tuple[str, str]]) -> None:
    """排一個 timer,把接下來彈出的 modal 對話框內容記下來再關掉。

    offscreen 測試不能讓 ``QMessageBox.exec()`` 卡死,但又要驗它真的彈了、內容對不對。
    ``singleShot(0)`` 排的 timer 會在對話框自己的 event loop 第一輪跑到;若對話框根本
    沒彈,timer 只是空跑一次,不會卡住也不會誤報。
    """

    def grab() -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            return
        store.append((dialog.windowTitle(), getattr(dialog, "text", lambda: "")()))
        dialog.accept()

    QTimer.singleShot(0, grab)


def catch_dialog(store: list, accept: bool = False) -> None:
    """抓下一個跳出來的 modal 對話框,把**物件本身**記下來再關掉。

    與 ``catch_modal`` 的差別在存什麼:自訂對話框(DeleteTrackDialog)要驗的是
    裡面的欄位值與按鈕狀態,不是一句 text()。物件的 parent 是主視窗,關掉之後
    仍然活著,所以關完再檢查沒問題。

    ``accept=True`` 走「按下確定」那條路。QDialog.accept() 讓 exec() 回傳
    Accepted,與真的點下按鈕同一條路——這點與 QMessageBox 不同(那邊 accept()
    回傳的不是 StandardButton,等同取消)。
    """

    def grab() -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            return
        store.append(dialog)
        if accept:
            dialog.accept()
        else:
            dialog.reject()

    QTimer.singleShot(0, grab)


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
    source = Path(
        sys.argv[1] if len(sys.argv) > 1
        else r"D:\ws\detect_stream\out\gt_per_frames_0625_145125\000-020s"
    )
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
        # 要驗的是「掃描沒漏檔」,拿磁碟上的 json 檔數當期望值;寫死幀數會綁死
        # 特定樣本,換一份資料就報出與程式無關的假 FAIL。
        expected_frames = len(list((work / "labels").glob("*.json")))
        check(window.open_root(work), "open_root 成功")
        check(len(window.frames) == expected_frames,
              f"labels 有 {expected_frames} 個 json 就載入 {expected_frames} 幀"
              f"(實際 {len(window.frames)})")
        canvas = window.canvas
        app.processEvents()
        check(canvas.width() > 400 and canvas.height() > 300,
              f"canvas 有實際尺寸 {canvas.width()}x{canvas.height()}")
        check(canvas.frame is not None and canvas.frame.size == (3840, 1920),
              "首幀尺寸 3840x1920")

        section("環景 hit-test:接縫另一側點得到同一個框")
        # 造一個確定跨接縫的框,再從畫面另一側點它。canvas 若只測主圈就會漏掉。
        canvas.tf.wrap_x = True
        canvas.tf.zoom = canvas.tf.fit_zoom(canvas.size())
        canvas.tf.off_x = 0.0
        canvas.tf.off_y = 0.0
        window.frames[window.index].dets.append(
            Det(label="person", track_id=777, ppe="ng", bbox=[0.97, 0.30, 1.02, 0.45])
        )
        canvas.reload_dets()
        app.processEvents()
        seam_det = len(window.frames[window.index].dets) - 1
        # 框的右段繞回畫面左邊:x=1.00 那一圈 → widget x = 0.00*span,取框中段
        left_x = round(0.01 * canvas.tf.span_x)
        mid_y = round(canvas.tf.n2v(0.0, 0.375).y())
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(left_x, mid_y))
        app.processEvents()
        check(canvas.selected_index == seam_det,
              f"從畫面左緣點到繞過來的那一段,選中同一個框"
              f"(預期 index {seam_det},實際 {canvas.selected_index})")
        del window.frames[window.index].dets[seam_det]
        canvas.reload_dets()

        # _shifts() 必須比 visible_shifts 兩端各多一圈。visible_shifts 只答「哪幾圈的
        # 影像與視窗相交」,而框不一樣:寬度最多一整圈,且拖曳中的框 x1 可以為負、
        # x2 可以超過 1(環景下 x 夾限被移除),繞回來的那一段因此落在影像圈之外。
        # 少了低端那一圈 → 往右拖過接縫時繞回段消失;少了高端 → 往左拖時消失。
        # 這一條直接盯 _shifts() 的範圍:visible_shifts 的數學修前修後都一樣,
        # 只驗它分不出「擴一端」與「擴兩端」的差別。
        base = list(canvas.tf.visible_shifts(float(canvas.width())))
        got = list(canvas._shifts())
        check(got == list(range(base[0] - 1, base[-1] + 2)),
              f"_shifts() 兩端各多一圈:visible_shifts={base} → _shifts()={got}")

        # 收尾以 model 為準恢復檢視狀態,**不要**寫死 False:Task 6 之後 3840x1920
        # 會自動進環景,寫死 False 會讓後續區段的環景斷言假 FAIL。
        canvas.tf.wrap_x = window.frames[window.index].wrap_x
        canvas.tf.clamp_offset(canvas.size())
        app.processEvents()

        section("環景模式")
        check(window.act_wrap.isChecked(), "3840x1920 開檔自動進環景模式")
        check(canvas.tf.wrap_x, "畫布的檢視也是環景")
        check(window.frames[0].wrap_x, "model 的存檔語意也是環景")
        check("ERP" in window.lbl_wrap.text(), f"狀態列顯示模式 {window.lbl_wrap.text()!r}")

        # 環景下 pan 可以一直往旁邊拖:拖三次都不該停在同一個位置
        before_off = canvas.tf.off_x
        for _ in range(3):
            canvas.tf.pan_by(-400.0, 0.0)
            canvas.tf.clamp_offset(canvas.size())
        check(canvas.tf.off_x != before_off, "環景下水平 pan 不會被夾住不動")
        check(0.0 <= canvas.tf.off_x < canvas.tf.span_x,
              f"off_x 被取模回 [0, span_x) {canvas.tf.off_x}")
        shifts = list(canvas.tf.visible_shifts(float(canvas.width())))
        check(len(shifts) >= 1, f"visible_shifts 至少一圈 {shifts}")

        # 在接縫上新畫一個框:跨越 x=1.0 後應該存成延伸表示,不是兩截也不是巨框
        window._goto(0)
        app.processEvents()
        canvas.tf.wrap_x = True
        canvas.tf.zoom = canvas.tf.fit_zoom(canvas.size())
        canvas.tf.off_x = canvas.width() / 2.0 - canvas.tf.span_x
        canvas.tf.clamp_offset(canvas.size())
        app.processEvents()
        seam_x = round(canvas.tf.n2v(1.0, 0.0).x() - canvas.tf.span_x)
        n_before = len(window.frames[0].dets)
        drag(canvas, QPoint(seam_x - 25, 300), QPoint(seam_x + 25, 380))
        app.processEvents()
        check(len(window.frames[0].dets) == n_before + 1,
              f"接縫上畫出一個新框({n_before} → {len(window.frames[0].dets)})")
        if len(window.frames[0].dets) == n_before + 1:
            bbox = window.frames[0].dets[-1].bbox
            width = bbox[2] - bbox[0]
            check(0.0 < width < 0.5,
                  f"跨接縫的新框寬度合理、沒有被拉成橫跨整張圖 {bbox}")
            check(0.0 <= bbox[0] < 1.0, f"x1 落在 [0,1) {bbox[0]}")
            # 標題寫的是「應該存成延伸表示」,這條才是真的驗到那件事——沒有它,
            # 一個完全畫在 [0,1] 內、從未跨越接縫的框也會滿足上面兩條斷言,整段
            # 名不符實。x1 那條在 wrap 下近乎恆真(canonical_bbox 本來就把 x1 取模
            # 回 [0,1)),鑑別力在這裡。
            check(bbox[2] > 1.0, f"x2 越過 1.0,存成延伸表示 {bbox}")
            # 立刻清掉剛畫的框:留著的話,下面「關掉環景」只是想測開關本身,卻會
            # 因為這個跨縫框撞上 Task 6 的確認對話框(有跨縫框時關環景會彈
            # QMessageBox,offscreen 測試沒有人按按鈕,會卡死在 modal 上)。
            del window.frames[0].dets[-1]
            canvas.reload_dets()

        # 貼邊警示:造一個 x2 恰好 1.0 的框,幀清單那一列要出現 CUT。斷言比的是
        # 摘要數字的**變化量**,不是寫死「1 幀」——樣本本身可能已經有貼邊框(例如
        # 收尾檢查要開的 100-120s 那種段落,那邊本來就有 94 幀貼邊),寫死絕對值
        # 換一份資料就假 FAIL,正是「驗收不得寫死資料量」要擋的事。
        def edge_frames_in_summary() -> int:
            m = re.search(r"貼邊 (\d+) 幀", window.list_panel.summary.text())
            return int(m.group(1)) if m else -1

        window._goto(0)
        app.processEvents()
        frame0 = window.frames[0]
        before_edge = edge_frames_in_summary()
        row0_had_cut = "CUT" in window.list_panel.list.item(0).text()
        frame0.dets.append(
            Det(label="person", track_id=778, ppe="ng", bbox=[0.96, 0.30, 1.0, 0.45])
        )
        window.list_panel.refresh_row(0, frame0)
        window.list_panel.refresh_summary(window.frames)
        app.processEvents()
        check(frame0.has_edge_box, "x2 恰好 1.0 → model 判定為貼邊")
        check("CUT" in window.list_panel.list.item(0).text(),
              f"幀清單那一列出現 CUT 旗標 {window.list_panel.list.item(0).text()!r}")
        check(edge_frames_in_summary() == before_edge + 1,
              f"摘要的貼邊幀數 +1({before_edge} → {edge_frames_in_summary()})")
        del frame0.dets[-1]
        canvas.reload_dets()
        window.list_panel.refresh_row(0, frame0)
        window.list_panel.refresh_summary(window.frames)
        check(edge_frames_in_summary() == before_edge, "移除後貼邊幀數回到原值")
        check(("CUT" in window.list_panel.list.item(0).text()) == row0_had_cut,
              "移除後那一列的 CUT 回到原本的狀態")

        # 關掉環景:含跨縫框的幀變成未存(存出去確實會不同)。資料集裡若還有別的
        # 跨縫框(不限剛畫的那個 —— 換一份本來就含跨縫框的樣本來跑也一樣),Task 6
        # 的確認對話框會彈出來;這裡按「是」模擬真人確認。沒有跨縫框時它不會彈,
        # 這個 timer 只是空跑一次,不影響任何斷言。
        def _confirm_leave_wrap() -> None:
            dialog = QApplication.activeModalWidget()
            if dialog is None:
                return
            yes = dialog.button(QMessageBox.StandardButton.Yes)
            if yes is not None:
                yes.click()
            else:
                dialog.accept()

        QTimer.singleShot(0, _confirm_leave_wrap)
        window.act_wrap.setChecked(False)
        app.processEvents()
        # offscreen 平台關掉 modal 後不會把 active window 還給主視窗,焦點與快捷鍵
        # 都只在 active window 裡成立——同一招用在別的 modal 上(見「幀清單多選 +
        # Delete」那幾段),這裡若真的彈了對話框也要補一次,否則後面每一段按鍵測試
        # 都會靜默落空。沒彈的話這兩行是沒有作用的空操作。
        window.activateWindow()
        app.processEvents()
        check(not canvas.tf.wrap_x, "取消勾選後畫布回到非環景")
        check(not window.frames[0].wrap_x, "model 的存檔語意也跟著回去")
        # 載入時每一幀的 wrap_x 預設就是 True(equirect 判定):只看 frames[0]
        # 驗不出「_apply_wrap_mode 忘了遍歷別的幀」這種 bug——有這種 bug 的版本
        # 只改 frames[0],frames[1..] 從沒被碰過,原本的預設值剛好把漏洞遮起來,
        # 一定要在 OFF 狀態逐一檢查每一幀才有鑑別力。
        check(not any(f.wrap_x for f in window.frames),
              "關掉後每一幀的存檔語意都跟著關(不只當前幀)")
        check(window.lbl_wrap.text() == "", "狀態列的環景字樣消失")

        # 翻幀也要在 OFF 狀態驗一次:有 bug 的版本只改了 frames[0],翻到別的幀
        # 時 canvas.set_frame 會用那幀從沒被改過的 wrap_x(還是 True)覆寫畫布,
        # 這裡才抓得到「模式被悄悄帶回來」。開回環景後翻幀對同一個 bug 是盲的,
        # 因為載入時的預設值本來就是 True,關掉再開回來等於把所有幀還原成那個
        # 預設值。
        for _ in range(3):
            window._navigate(1)
            app.processEvents()
        check(not canvas.tf.wrap_x,
              "關掉後翻幾幀仍是非環景(set_frame 不會從別的幀把模式帶回來)")
        window._goto(0)
        app.processEvents()

        window.act_wrap.setChecked(True)
        app.processEvents()
        check(canvas.tf.wrap_x, "重新勾選回到環景")

        # _apply_wrap_mode 為每一幀寫 wrap_x 的唯一理由:canvas.set_frame 會用
        # 該幀的 wrap_x 覆寫檢視狀態,少寫一幀就會在切到它時悄悄還原模式。ON 方向
        # 對「漏寫某些幀」是盲的(理由同上,見 OFF 狀態那一組),留著是為了兩個
        # 方向都測到「翻幀不會被 set_frame 干擾」這個機制本身。
        for _ in range(3):
            window._navigate(1)
            app.processEvents()
        check(canvas.tf.wrap_x, "翻幾幀之後仍在環景模式(模式沒有被 set_frame 還原)")
        check(all(f.wrap_x for f in window.frames), "每一幀的存檔語意都是環景")
        window._goto(0)
        app.processEvents()

        section("非 2:1 合成資料不自動進環景")
        # 「非 2:1 行為完全不變」這條 constraint 在 GUI 層還沒有任何覆蓋。自己造
        # 幾張 1920x1080 的假標註驗一次 —— 這段會換掉開著的資料夾,驗完換回
        # work,否則後面每一段都會對著這份假資料跑。
        flat_dir = tmp / "gt_flat"
        (flat_dir / "frames").mkdir(parents=True)
        (flat_dir / "labels").mkdir(parents=True)
        for i in range(3):
            (flat_dir / "labels" / f"{i:06d}.json").write_text(
                json.dumps({
                    "type": "gt", "version": 1, "seq": i + 1, "frame_index": i,
                    "video_sec": float(i), "size": [1920, 1080],
                    "image": f"../frames/{i:06d}.jpg",
                    "dets": [
                        {"label": "person", "track_id": 1, "ppe": "ng",
                         "bbox": [0.1, 0.1, 0.2, 0.2]},
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        check(window.open_root(flat_dir), "開啟 1920x1080 的假資料")
        check(not window.act_wrap.isChecked(), "非 2:1 不自動進環景")
        check(not canvas.tf.wrap_x, "畫布也不是環景")
        check(window.lbl_wrap.text() == "", "狀態列不顯示環景字樣")

        check(window.open_root(work), "換回原本的 equirect 資料夾")
        app.processEvents()

        # 逐幀瀏覽測的是每幀解碼成本,走幾幀夠了就好;整份 600 幀走完只是把
        # 同一件事重複八次,徒增驗收時間。
        walk = min(WALK_FRAMES, len(window.frames) - 1)
        section(f"逐幀瀏覽 {walk + 1} 幀")
        t0 = time.perf_counter()
        visited = 0
        for _ in range(walk):
            window._navigate(1)
            app.processEvents()
            visited += 1
        first_pass = time.perf_counter() - t0
        check(window.index == walk, f"連續前進到 index {walk}(實際 {window.index})")
        check(visited == walk, f"{walk} 次前進全部成功")
        print(f"       首輪(含 JPEG 解碼)共 {first_pass * 1000:.0f} ms,"
              f"平均 {first_pass / walk * 1000:.1f} ms/幀")

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

        section("視窗尺寸不被面板內容左右")
        # QMainWindow 會在 minimumSizeHint 超過當前尺寸時自己把視窗頂大,而且只長
        # 不縮。右側面板的最小高度若直接往上傳,選一個框(det_panel 要多顯示三行
        # bbox 座標)就會把視窗撐高、底部推出螢幕 —— 在高 DPI 縮放下必然發生,
        # 而使用者調好的視窗大小不該被「現在顯示了什麼」決定。
        window.resize(1280, 760)
        app.processEvents()
        size_before = (window.width(), window.height())
        hint_before = window.minimumSizeHint()
        probe_frame = window.frames[window.index]
        for k in range(min(6, len(probe_frame.dets))):
            canvas.select(k)
            app.processEvents()
        canvas.select(-1)
        app.processEvents()
        check((window.width(), window.height()) == size_before,
              f"連選 6 個框後視窗尺寸不變(實際 {size_before} -> "
              f"{(window.width(), window.height())})")
        check(window.minimumSizeHint() == hint_before,
              f"視窗最小尺寸也不隨內容變({hint_before} -> {window.minimumSizeHint()})")
        # 最小高度要能塞進小螢幕:150% DPI 縮放的 1080p 邏輯高度只有 720。
        check(window.minimumSizeHint().height() <= 720,
              f"視窗最小高度 {window.minimumSizeHint().height()} <= 720,"
              f"高 DPI 螢幕塞得下")

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
        # track_id 的預設是「沿用」,而前面已經點過框(候選有值)。這兩段要測的是
        # 自動取號,得先明確切過去,否則新框會掛上剛剛點過的那個號碼。
        window.new_panel.auto_radio.setChecked(True)
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
        check(window.new_panel.follow_id() is None, "選了自動時 follow_id() 回 None")

        # 補錨點是主力用法,面板一開起來就該站在「沿用」上、而且排在「自動」前面。
        # 拿全新的面板來驗:window 上那個已經被前面幾段切來切去。
        fresh = NewDetPanel()
        fresh_box = fresh.follow_radio.parentWidget().layout()
        check(fresh.follow_radio.isChecked(), "「沿用」是 track_id 的開機預設")
        check(fresh_box.indexOf(fresh.follow_radio) < fresh_box.indexOf(fresh.auto_radio),
              "「沿用」排在「自動」前面")
        check(fresh.follow_id() is None, "還沒點過框時「沿用」退回自動取號(follow_id() 回 None)")
        fresh.deleteLater()

        anchor_pos = next(k for k, d in enumerate(frame.dets) if d.track_id is not None)
        canvas.select(anchor_pos)
        app.processEvents()
        anchor_id = frame.dets[anchor_pos].track_id
        check(f"#{anchor_id}" in window.new_panel.follow_radio.text(),
              f"點過框後 radio 顯示候選號碼:{window.new_panel.follow_radio.text()!r}")

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
        window._delete_selected()
        app.processEvents()

        section("面板編輯 + Delete")
        window.det_panel.track_edit.setText("777")
        window.det_panel.track_edit.editingFinished.emit()
        app.processEvents()
        check(frame.dets[-1].track_id == 777, f"track_id 寫入 777(實際 {frame.dets[-1].track_id})")

        # 改號後「沿用 #id」必須跟著換:停在舊號的話,接著補的錨點會默默掛回剛
        # 淘汰掉的號碼,而畫面上沒有任何跡象。
        check("#777" in window.new_panel.follow_radio.text(),
              f"改 id 後沿用候選顯示 #777(實際 {window.new_panel.follow_radio.text()!r})")
        window.new_panel.follow_radio.setChecked(True)
        check(window.new_panel.follow_id() == 777,
              f"改 id 後 follow_id() = {window.new_panel.follow_id()}(預期 777)")
        window.new_panel.auto_radio.setChecked(True)

        window.det_panel.label_box.setCurrentText("drone")
        app.processEvents()
        check(frame.dets[-1].label == "drone", "label 改成 drone")
        check(not window.det_panel.ppe_box.isEnabled(), "drone 的 ppe 欄位被鎖住")

        window.det_panel.label_box.setCurrentText("person")
        window.det_panel.ppe_box.setCurrentIndex(window.det_panel.ppe_box.findData("ng"))
        app.processEvents()
        check(frame.dets[-1].ppe == "ng", f"ppe 寫入 ng(實際 {frame.dets[-1].ppe})")

        # 刪除不能動到黏著記憶:刪完選取會遞補到隔壁的框,那是 index 位移的副作用,
        # 不是使用者指定的軌跡。飄掉的話,接著補的錨點會默默掛上隔壁那條的號碼。
        #
        # 挑一個「下一個框屬於別條軌跡」的位置來刪,這條驗收才有鑑別力:遞補到同
        # 一條的話,記憶飄不飄都看不出差別。刪掉 pos 之後選取會落回 pos,也就是
        # 原本 pos+1 的那個框。
        pos = next((k for k in range(len(frame.dets) - 1)
                    if frame.dets[k].track_id is not None
                    and frame.dets[k + 1].track_id != frame.dets[k].track_id), None)
        check(pos is not None, "找到一個刪掉後會遞補到別條軌跡的框")
        if pos is not None:
            canvas.select(pos)
            app.processEvents()
            neighbour_id = frame.dets[pos + 1].track_id
            check(f"#{neighbour_id}" not in window.new_panel.follow_radio.text(),
                  f"刪之前沿用停在 {window.new_panel.follow_radio.text()!r},"
                  f"與會遞補上來的 #{neighbour_id} 不同")
        follow_before = window.new_panel.follow_radio.text()
        target_before = window._last_track
        count_before = len(frame.dets)
        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        app.processEvents()
        check(len(frame.dets) == count_before - 1, f"Delete 刪掉一框 -> {len(frame.dets)}")
        check(window.new_panel.follow_radio.text() == follow_before,
              f"刪框後沿用候選不變({follow_before!r} → "
              f"{window.new_panel.follow_radio.text()!r})")
        check(window._last_track == target_before,
              f"刪框後刪除目標不變({target_before} → {window._last_track})")

        section("復原/重做 25 步")
        baseline = frame.dets_json()
        canvas.select(0)
        for i in range(25):
            # 每步都給一個唯一、且不貼邊的框。不能用「往右下累加」:累加的基礎是
            # 上一步 clamp 後的值,框會被推到 [1,1,1,1] 飽和,之後的編輯不再改變
            # 狀態,而 UndoStack 對「狀態沒變」依設計不記一筆(commit 回 False),
            # 25 次編輯只留 24 筆歷史,25 次 undo 就退過頭一步。飽和發生在第幾步
            # 取決於框原本離邊界多遠,所以那種寫法會隨資料集翻臉。
            span = 0.10 + 0.001 * i
            frame.dets[0].bbox = [0.20, 0.20, 0.20 + span, 0.20 + span]
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
        # 環景模式下跨縫框的 x2 本來就該越過 1.0,寫死 [0,1] 會在資料集出現真正的
        # 跨縫框時假 FAIL。判準與 verify_roundtrip.test_edit_roundtrip 同一套。
        wrap = reopened.wrap_x
        check(all(0.0 <= d["bbox"][0] < 1.0 for d in reopened.dets_json()),
              f"重開後 x1 都在 [0,1)(環景={wrap})")
        check(all(0.0 <= d["bbox"][1] <= 1.0 and 0.0 <= d["bbox"][3] <= 1.0
                  for d in reopened.dets_json()),
              "重開後 y 都在 [0,1]")

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
            # 期望值直接取 interpolate_missing 的輸出,驗的是「apply_interpolation 真的把
            # 算出來的框寫進了對應的幀」——那才是這一層(GUI)該負責的事。
            # 內插本身的數學正確性由 tests/verify_roundtrip.py 的單元測試負責,含環景下
            # 走最短弧的案例;在這裡重算一遍等於把實作抄進測試,實作一改它就過時
            # (equirect 的最短弧就是這樣讓舊版本假 FAIL 的)。
            want = {i: det.bbox for i, det in loose.additions}
            exact = all(filled[i].bbox == want[i] for i in hole)
            check(exact, f"補回的每個 bbox 都等於 interpolate_missing 算出的值"
                         f"({len(hole)} 幀)")

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

        section("幀清單多選 + Delete:刪掉選取範圍內的整條 track")
        frame_list = window.list_panel.list
        check(frame_list.selectionMode() ==
              QAbstractItemView.SelectionMode.ExtendedSelection,
              "幀清單可多選(Shift 拉連續 / Ctrl 加點零散)")
        check(any(QKeySequence(Qt.Key.Key_Delete) in action.shortcuts()
                  for action in frame_list.actions()),
              "Delete 綁在幀清單本身,不搶畫布的「刪掉這一個框」")

        check(window.act_delete_track in window.menuBar().actions()[1].menu().actions(),
              "「刪除選取幀內的軌跡…」收在「編輯」選單裡")
        check(window.act_delete_track.shortcut() == QKeySequence("Ctrl+Shift+D"),
              f"快捷鍵 Ctrl+Shift+D(實際 {window.act_delete_track.shortcut().toString()})")
        check(window.list_panel.minimumSizeHint().height() < 300,
              f"左欄沒有被常駐面板墊高(minHint 高 "
              f"{window.list_panel.minimumSizeHint().height()})")

        # 沒選幀就不該跳對話框,而且要說明原因而非靜默不動。
        frame_list.clearSelection()
        counts_before = [len(f.dets) for f in window.frames]
        window._delete_track_range()
        app.processEvents()
        check([len(f.dets) for f in window.frames] == counts_before,
              "沒選任何幀時按 Delete 不動資料")
        check("幀清單" in window.statusBar().currentMessage(),
              f"且有說明原因:{window.statusBar().currentMessage()!r}")

        # 最後點過的框只是對話框的**預填值**,不是背後偷偷送出去的執行參數。
        window._goto(rows[0])
        canvas.select(next(k for k, d in enumerate(window.frames[rows[0]].dets)
                           if on_track(d)))
        app.processEvents()
        check(window._last_track == (probe_label, probe_tid),
              f"點框後記住 {probe_label} #{probe_tid}(實際 {window._last_track})")
        window._goto(rows[-1])
        app.processEvents()
        check(canvas.selected_det is None, "切幀後畫布選取被清掉(所以問不到目標)")
        check(window._last_track == (probe_label, probe_tid), "但記住的軌跡跨幀存活")

        # 真的用 Shift+↓ 拉範圍,不是直接呼叫 API 設選取 —— 要驗的正是這條路。
        # 拉幾列要設上限:每按一次就切一幀、跟著解一張 3840x1920 的 JPEG,而
        # per-frame GT 的軌跡動輒橫跨全部 600 幀,拉半條就要等上好幾十秒。要驗的是
        # 「Shift 能不能拉出連續範圍」,幾十列就夠;上限只縮短範圍,不影響任何斷言
        # (範圍外剩的框只會更多)。
        del_lo = rows[0]
        del_hi = min(rows[len(rows) // 2 - 1], del_lo + SHIFT_RANGE_ROWS)
        frame_list.setFocus()
        frame_list.setCurrentRow(del_lo)
        app.processEvents()
        for _ in range(del_hi - del_lo):
            QTest.keyClick(frame_list, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        app.processEvents()
        check(window.list_panel.selected_rows() == list(range(del_lo, del_hi + 1)),
              f"Shift+↓ 拉出連續範圍 {del_lo}..{del_hi}"
              f"(實際 {window.list_panel.selected_rows()[:3]}… 共 "
              f"{len(window.list_panel.selected_rows())} 列)")
        check(window.index == del_hi, "拉範圍時當前幀跟著走到範圍末端(語意不變)")

        doomed_rows = [i for i in rows if del_lo <= i <= del_hi]
        kept_rows = [i for i in rows if i > del_hi]
        check(bool(doomed_rows) and bool(kept_rows),
              f"範圍內 {len(doomed_rows)} 幀要刪、範圍外 {len(kept_rows)} 幀要留")

        # 人在清單裡操作時 _goto 不搶焦點(搶了 ↑↓ 與 Delete 就全落空),所以焦點會
        # 一直留在清單上。工具列那些 window 層快捷鍵必須照樣到得了 —— 真實環境的
        # 按鍵一律送到焦點 widget,這裡就照那樣送。
        check(frame_list.hasFocus(), "拉完範圍焦點仍在幀清單(↑↓ 與 Delete 才有效)")
        QTest.keyClick(frame_list, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        check(window.find_edit.hasFocus(), "焦點在幀清單時 Ctrl+F 照樣聚焦搜尋欄")
        window.find_edit.clear()
        window.find_kind.setCurrentIndex(0)
        frame_list.setFocus()

        # 同號但不同 label 的框:認軌是 (label, track_id),它絕對不能被一起刪掉。
        other_label = "drone" if probe_label == "person" else "person"
        ghost_row = doomed_rows[0]
        ghost = Det(label=other_label, track_id=probe_tid, ppe=None,
                    bbox=[0.90, 0.10, 0.95, 0.15])
        window.frames[ghost_row].dets.append(ghost)

        plan = plan_track_delete(window.frames, probe_label, probe_tid,
                                 window.list_panel.selected_rows())
        check(plan.frame_indexes == doomed_rows,
              f"算出要刪 {len(doomed_rows)} 幀(實際 {len(plan.frame_indexes)})")
        outside = sum(1 for i in kept_rows for d in window.frames[i].dets if on_track(d))
        check(plan.outside == outside,
              f"回報範圍外還有 {outside} 個框會保留(實際 {plan.outside})")
        check(all(d is not ghost for _, d in plan.targets),
              f"同號的 {other_label} #{probe_tid} 不在刪除清單")

        # 取消那條路:對話框跳出來、內容對,按取消就什麼都不動。
        counts_now = [len(f.dets) for f in window.frames]
        opened: list = []
        catch_dialog(opened)
        QTest.keyClick(frame_list, Qt.Key.Key_Delete)
        app.processEvents()
        # offscreen 平台關掉 modal 後不會把 active window 還給主視窗(activeWindow()
        # 變 None),而焦點與快捷鍵都只在 active window 裡成立 —— 不補這一下,後面
        # 每一段的按鍵測試都會靜默落空。這是測試環境的還原,不是產品行為。
        window.activateWindow()
        app.processEvents()
        check(len(opened) == 1 and opened[0].windowTitle() == "刪除軌跡片段",
              f"在幀清單按 Delete 跳出對話框(實際 {[d.windowTitle() for d in opened]})")
        if opened:
            dialog = opened[0]
            check(dialog.target() == (probe_label, probe_tid),
                  f"預填最後點過的那條 {probe_label} #{probe_tid}"
                  f"(實際 {dialog.target()})")
            ok_button = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
            check(f"{plan.box_count}" in ok_button.text(),
                  f"按鈕上就寫著要刪幾個框:{ok_button.text()!r}")
            check(f"命中 {len(doomed_rows)} 幀" in dialog.detail.text()
                  and f"範圍外還有 {outside} 個框" in dialog.detail.text(),
                  f"寫出命中多少、範圍外還留著多少:"
                  f"{dialog.detail.text().replace(chr(10), ' | ')!r}")

            # 即時連動:改成一個沒人用的號碼,按鈕當場停用,不必先送出才知道白做。
            nobody = max(d.track_id for f in window.frames for d in f.dets
                         if d.track_id is not None) + 777
            dialog.track_edit.setText(str(nobody))
            app.processEvents()
            check(not ok_button.isEnabled(),
                  f"改成查無此人的 #{nobody} 後「確定」停用")
            check("沒有" in dialog.detail.text(),
                  f"並說明為什麼:{dialog.detail.text()!r}")
        check([len(f.dets) for f in window.frames] == counts_now,
              "按取消時不動資料")

        # 確定那條路:走完整流程(對話框 accept)而不是直接呼叫 apply。
        follow_before = window.new_panel.follow_radio.text()
        accepted: list = []
        catch_dialog(accepted, accept=True)
        QTest.keyClick(frame_list, Qt.Key.Key_Delete)
        app.processEvents()
        window.activateWindow()
        app.processEvents()
        check(len(accepted) == 1, "第二次按 Delete 又跳出對話框")
        check(not any(on_track(d) for i in doomed_rows for d in window.frames[i].dets),
              f"選取範圍內的 {probe_label} #{probe_tid} 全部消失")
        check(all(any(on_track(d) for d in window.frames[i].dets) for i in kept_rows),
              f"範圍外的 {len(kept_rows)} 幀原封不動")
        check(any(d is ghost for d in window.frames[ghost_row].dets),
              f"同號的 {other_label} #{probe_tid} 沒被波及(認軌是 (label, track_id))")
        check(window.frames[ghost_row].dirty, "刪完該幀標記未存")
        check(window._last_track == (probe_label, probe_tid),
              "刪完預填值仍停在同一條(幽靈框常分好幾段,要能接著刪下一段)")
        # 回歸錨點,不是鑑別測試:實測拿掉 _remember_track 的凍結後這條仍會通過
        # ——這個樣本刪完剛好遞補回同一條。真正抓得到凍結失效的是上面「面板編輯 +
        # Delete」那段(它刻意挑了會遞補到別條軌跡的位置)。留著是防 apply_track_delete
        # 之後被改動時誤傷。
        check(window.new_panel.follow_radio.text() == follow_before,
              f"刪完「沿用 #id」不變({follow_before!r} → "
              f"{window.new_panel.follow_radio.text()!r})")
        check(canvas.hasFocus(), "刪完焦點交還畫布(接著就是看畫面確認,A / D 也還能翻)")

        window._undo_last_bulk()
        app.processEvents()
        check(all(any(on_track(d) for d in window.frames[i].dets) for i in doomed_rows),
              "Ctrl+Shift+I 整組復原,刪掉的框全部回來")

        # 選單入口:焦點不在幀清單時 Delete 鍵到不了清單,選單那條路必須也能開。
        canvas.setFocus()
        app.processEvents()
        check(not frame_list.hasFocus(), "焦點在畫布上(Delete 鍵此時是刪單一個框)")
        via_menu: list = []
        catch_dialog(via_menu)
        window.act_delete_track.trigger()
        app.processEvents()
        check(len(via_menu) == 1 and via_menu[0].windowTitle() == "刪除軌跡片段",
              f"從「編輯」選單也叫得出同一個對話框"
              f"(實際 {[d.windowTitle() for d in via_menu]})")
        window.activateWindow()
        app.processEvents()

        # 還原成磁碟上的樣子:拿掉造出來的框。復原是用 clone 寫回去的,原本那顆
        # ghost 物件已經不在 dets 裡,只能用值比對。
        window.frames[ghost_row].dets[:] = [
            d for d in window.frames[ghost_row].dets
            if not (d.label == other_label and d.track_id == probe_tid
                    and d.bbox == [0.90, 0.10, 0.95, 0.15])
        ]
        check(not any(window.frames[i].dirty for i in doomed_rows),
              "還原後這幾幀回到已存狀態")
        frame_list.clearSelection()

        section("只改選取幀內的 track_id")
        check(window.act_remap_range.shortcut() == QKeySequence("Ctrl+Shift+R"),
              f"快捷鍵 Ctrl+Shift+R"
              f"(實際 {window.act_remap_range.shortcut().toString()})")
        check(window.act_remap_range in window.menuBar().actions()[1].menu().actions(),
              "「改選取幀內的軌跡 id…」收在「編輯」選單裡")

        frame_list.clearSelection()
        ids_untouched = [(d.label, d.track_id) for f in window.frames for d in f.dets]
        window._remap_range()
        app.processEvents()
        check([(d.label, d.track_id) for f in window.frames for d in f.dets]
              == ids_untouched, "沒選任何幀時不動資料")
        check("幀清單" in window.statusBar().currentMessage(),
              f"且有說明原因:{window.statusBar().currentMessage()!r}")

        # 圈出同一段,把它改成一個沒人用的號碼 —— 這正是 tracker 把某一段誤配給
        # 別的目標時要做的事:只拆那一段,其餘維持原號。
        frame_list.setFocus()
        frame_list.setCurrentRow(del_lo)
        app.processEvents()
        for _ in range(del_hi - del_lo):
            QTest.keyClick(frame_list, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        app.processEvents()
        spare_id = max(d.track_id for f in window.frames for d in f.dets
                       if d.track_id is not None) + 500
        inside_before = sum(1 for i in doomed_rows
                            for d in window.frames[i].dets if on_track(d))
        outside_before = sum(1 for i in kept_rows
                             for d in window.frames[i].dets if on_track(d))

        ranged = plan_remap(window.frames, probe_label, probe_tid, spare_id,
                            window.list_panel.selected_rows())
        check(ranged.box_count == inside_before,
              f"算出只改範圍內的 {inside_before} 個框(實際 {ranged.box_count})")
        check(ranged.outside == outside_before,
              f"回報範圍外還有 {outside_before} 個框留著舊號(實際 {ranged.outside})")
        check(not ranged.conflicts,
              f"目標號沒人用 → 沒有撞號警告(實際 {ranged.conflicts})")
        # 不帶範圍就該是全域,既有呼叫端的行為一個字都不能變。
        whole = plan_remap(window.frames, probe_label, probe_tid, spare_id)
        check(whole.box_count == inside_before + outside_before,
              f"不帶範圍仍是全域換號(實際 {whole.box_count} 個框)")
        check(whole.outside == 0 and whole.merges == 0,
              "全域模式沒有「範圍外」可言,兩個計數都是 0")

        opened_remap: list = []
        catch_dialog(opened_remap, accept=True)
        window.act_remap_range.trigger()
        app.processEvents()
        window.activateWindow()
        app.processEvents()
        check(len(opened_remap) == 1
              and opened_remap[0].windowTitle() == "改軌跡片段的 id",
              f"跳出對話框(實際 {[d.windowTitle() for d in opened_remap]})")
        if opened_remap:
            remap_dialog = opened_remap[0]
            check(remap_dialog.old_edit.text() == str(probe_tid),
                  f"舊號預填最後點過的那條(實際 {remap_dialog.old_edit.text()!r})")
            check(remap_dialog.new_edit.text() == "", "新號留白等人填")
        # 上面 accept 時新號還空著,target() 是 None —— 不能因此改壞任何資料。
        check([(d.label, d.track_id) for f in window.frames for d in f.dets]
              == ids_untouched, "新號沒填就按確定也不動資料")

        window.apply_remap(ranged, probe_label, probe_tid, spare_id)
        app.processEvents()
        check(all(not any(on_track(d) for d in window.frames[i].dets)
                  for i in doomed_rows),
              f"範圍內的 {probe_label} #{probe_tid} 全部換成 #{spare_id}")
        check(sum(1 for i in doomed_rows for d in window.frames[i].dets
                  if d.label == probe_label and d.track_id == spare_id)
              == inside_before,
              f"換過去的框數對得上({inside_before} 個)")
        check(all(any(on_track(d) for d in window.frames[i].dets) for i in kept_rows),
              f"範圍外的 {len(kept_rows)} 幀仍是 #{probe_tid},沒被波及")

        # 反過來算一次:此時範圍外已有一整段用著目標號,merges 要報得出來。
        back = plan_remap(window.frames, probe_label, spare_id, probe_tid,
                          window.list_panel.selected_rows())
        check(back.merges == outside_before,
              f"回報範圍外已有 {outside_before} 個框是 #{probe_tid},改完接成同一條"
              f"(實際 {back.merges})")

        window._undo_last_bulk()
        app.processEvents()
        check(all(any(on_track(d) for d in window.frames[i].dets) for i in doomed_rows),
              "Ctrl+Shift+I 整組復原,號碼全部改回來")
        check(not any(d.track_id == spare_id for f in window.frames for d in f.dets),
              f"整份資料集不再有 #{spare_id}")
        check(not any(window.frames[i].dirty for i in doomed_rows),
              "還原後這幾幀回到已存狀態")
        frame_list.clearSelection()

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

        section("id 重疊警示")
        dup_row = next(
            (i for i, f in enumerate(window.frames)
             if not f.has_duplicate_track and any(d.track_id is not None for d in f.dets)),
            None,
        )
        check(dup_row is not None, "找到一幀本來沒有 id 重疊的列當基準")
        if dup_row is not None:
            dup_frame = window.frames[dup_row]
            base = next(d for d in dup_frame.dets if d.track_id is not None)
            check("DUP" not in window.list_panel.list.item(dup_row).text(),
                  "沒有重疊的列不帶 DUP 標記")

            # 摘要驗「比造假之前多一幀」,不驗絕對數字:資料集本身就可能帶著重疊
            # (上游 tracker 的產物,不是壞資料),寫死數字換一份樣本就報假 FAIL。
            dup_before = sum(1 for f in window.frames if f.has_duplicate_track)
            twin = Det(label=base.label, track_id=base.track_id, ppe=base.ppe,
                       bbox=[0.40, 0.40, 0.45, 0.45])
            dup_frame.dets.append(twin)
            window.list_panel.refresh_row(dup_row, dup_frame)
            check(dup_frame.has_duplicate_track,
                  f"同幀出現兩個 {base.label}#{base.track_id} → 判定為 id 重疊")
            check("DUP" in window.list_panel.list.item(dup_row).text(),
                  f"該列文字含 DUP 標記:{window.list_panel.list.item(dup_row).text()!r}")
            window.list_panel.refresh_summary(window.frames)
            check(f"id 重疊 {dup_before + 1} 幀" in window.list_panel.summary.text(),
                  f"摘要多算一幀({dup_before} → {dup_before + 1}):"
                  f"{window.list_panel.summary.text().replace(chr(10), ' | ')!r}")

            # 跨 label 同號是正常的(person#1 與 drone#1 是兩條軌跡),不該警示。
            twin.label = "drone" if base.label == "person" else "person"
            twin.ppe = "ng" if twin.label == "person" else None
            check(not dup_frame.has_duplicate_track,
                  f"改成 {twin.label}#{base.track_id} 後不算重疊(認軌是 (label, track_id))")

            dup_frame.dets.remove(twin)
            window.list_panel.refresh_row(dup_row, dup_frame)
            window.list_panel.refresh_summary(window.frames)
            check(not dup_frame.dirty and "DUP" not in
                  window.list_panel.list.item(dup_row).text(),
                  "移除後標記消失且不算未存")

        section("找 track(Ctrl+F)")
        # 搜尋是唯讀動作:整段跑完不該讓任何一幀變成未存。
        check(not any(f.dirty for f in window.frames), "搜尋前所有幀都是已存狀態")
        hits = find_track(window.frames, probe_label, probe_tid)
        manual = [
            (i, k)
            for i, f in enumerate(window.frames)
            for k, d in enumerate(f.dets)
            if d.label == probe_label and d.track_id == probe_tid
        ]
        check(hits == manual, f"find_track 列出 {len(hits)} 次出現,與手算逐項一致")
        check(len(hits) >= 3, f"{probe_label}#{probe_tid} 至少出現 3 次(實際 {len(hits)})")

        # 認軌是 (label, track_id):同號但不同 label 的框絕不能被限定搜尋撈到。
        other = "drone" if probe_label == "person" else "person"
        decoy2 = Det(label=other, track_id=probe_tid,
                     ppe="ng" if other == "person" else None,
                     bbox=[0.80, 0.20, 0.85, 0.26])
        decoy2_frame = window.frames[hits[0][0]]
        decoy2_frame.dets.append(decoy2)
        check(find_track(window.frames, probe_label, probe_tid) == hits,
              f"多了 {other}#{probe_tid} 也不影響限定 {probe_label} 的搜尋結果")
        check(len(find_track(window.frames, None, probe_tid)) == len(hits) + 1,
              "選「全部」時同號的另一個 label 會被列入")
        decoy2_frame.dets.remove(decoy2)
        check(not decoy2_frame.dirty, "移除誘餌後該幀回到已存狀態")

        # Ctrl+F:真的按鍵。同時驗證它沒被畫布的 F(還原檢視)吃掉。
        first_i, first_k = hits[0]
        window._goto(first_i)
        canvas.select(first_k)
        window.find_edit.clear()
        window.find_kind.setCurrentIndex(0)
        canvas.setFocus()
        app.processEvents()
        off_fit_zoom = canvas.tf.zoom * 1.5
        canvas.tf.zoom = off_fit_zoom
        QTest.keyClick(canvas, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        check(window.find_edit.hasFocus(), "Ctrl+F 聚焦搜尋欄(快捷鍵真的接上)")
        check(abs(canvas.tf.zoom - off_fit_zoom) < 1e-9,
              "Ctrl+F 沒被畫布的 F 吃掉(縮放沒被打回 fit)")
        check(window.find_edit.text() == str(probe_tid),
              f"預填選取框的號碼(實際 {window.find_edit.text()!r})")
        check(window.find_kind.currentData() == probe_label,
              f"連 label 一起預填(實際 {window.find_kind.currentData()!r})")

        # Enter / Shift+Enter:焦點在搜尋欄時連按就能走完整條軌跡。
        QTest.keyClick(window.find_edit, Qt.Key.Key_Return)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[1],
              f"Enter 跳到第 2 次出現(實際 {(window.index, canvas.selected_index)},"
              f"預期 {hits[1]})")
        picked = canvas.selected_det
        check(picked is not None and picked.label == probe_label
              and picked.track_id == probe_tid,
              "跳過去後選中的正是該 track 的框")
        check(window.find_edit.hasFocus(), "跳完焦點留在搜尋欄,可以連按 Enter")
        check(f"/{len(hits)} 次出現" in window.statusBar().currentMessage(),
              f"訊息報出第幾次 / 共幾次:{window.statusBar().currentMessage()!r}")

        QTest.keyClick(window.find_edit, Qt.Key.Key_Return,
                       Qt.KeyboardModifier.ShiftModifier)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[0],
              f"Shift+Enter 退回第 1 次出現(實際 {(window.index, canvas.selected_index)})"
              f"——沒被 QLineEdit 的 returnPressed 當成往後找")

        # F3 / Shift+F3:焦點在畫布時也能繼續找,且不把焦點搬進輸入框。
        canvas.setFocus()
        app.processEvents()
        QTest.keyClick(canvas, Qt.Key.Key_F3)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[1],
              f"F3 找下一個(實際 {(window.index, canvas.selected_index)})")
        check(canvas.hasFocus(), "F3 不把焦點搬進搜尋欄(A/D 翻幀還能用)")
        QTest.keyClick(canvas, Qt.Key.Key_F3, Qt.KeyboardModifier.ShiftModifier)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[0], "Shift+F3 找上一個")

        # 掃到盡頭繞回另一端。
        window._goto(hits[-1][0])
        canvas.select(hits[-1][1])
        app.processEvents()
        window._find_next(1)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[0],
              f"最後一次出現處再找下一個 → 繞回第一次"
              f"(實際 {(window.index, canvas.selected_index)})")
        check("繞回" in window.statusBar().currentMessage(),
              f"且訊息註明繞回:{window.statusBar().currentMessage()!r}")
        window._find_next(-1)
        app.processEvents()
        check((window.index, canvas.selected_index) == hits[-1], "反向也會繞回最後一次")

        # 同幀兩個同號框(清單的 DUP)要逐個停,不能整幀跳過——第二個框才是要修的。
        dup_i = hits[0][0]
        twin2 = Det(label=probe_label, track_id=probe_tid,
                    ppe="ng" if probe_label == "person" else None,
                    bbox=[0.10, 0.60, 0.14, 0.70])
        window.frames[dup_i].dets.append(twin2)
        in_frame = [k for i, k in find_track(window.frames, probe_label, probe_tid)
                    if i == dup_i]
        check(len(in_frame) == 2, f"同幀兩個同號框列出 2 筆命中(實際 {len(in_frame)})")
        window._goto(dup_i)
        canvas.select(in_frame[0])
        app.processEvents()
        window._find_next(1)
        app.processEvents()
        check((window.index, canvas.selected_index) == (dup_i, in_frame[1]),
              f"停在同幀的第二個同號框(實際 {(window.index, canvas.selected_index)})")
        window.frames[dup_i].dets.remove(twin2)
        check(not window.frames[dup_i].dirty, "移除後該幀回到已存狀態")

        # 找不到要擋下來說清楚:狀態列閃一下容易被忽略,所以再彈一個警示視窗。
        stay = (window.index, canvas.selected_index)
        nobody = max(d.track_id for f in window.frames for d in f.dets
                     if d.track_id is not None) + 999
        window.find_edit.setText(str(nobody))
        popped: list[tuple[str, str]] = []
        catch_modal(popped)
        window._find_next(1)
        app.processEvents()
        check((window.index, canvas.selected_index) == stay, "找不到時位置不動")
        check(len(popped) == 1 and popped[0][0] == "找不到 track",
              f"找不到會彈警示視窗(實際 {popped})")
        missing_text = popped[0][1] if popped else ""
        check(f"沒有 {probe_label} #{nobody}" in missing_text,
              f"視窗寫出找不到誰:{missing_text!r}")
        check("label 下拉" not in missing_text,
              "這個號碼哪個 label 都沒有,就不亂給「選錯 label」的提示")
        check("找不到" in window.statusBar().currentMessage(),
              f"狀態列同時留一份:{window.statusBar().currentMessage()!r}")

        # 限定錯 label 是最常見的落空原因,視窗要直接指出號碼其實掛在哪一種。
        window.find_kind.setCurrentIndex(window.find_kind.findData(other))
        window.find_edit.setText(str(probe_tid))
        popped.clear()
        catch_modal(popped)
        window._find_next(1)
        app.processEvents()
        wrong_label = popped[0][1] if popped else ""
        check(f"#{probe_tid} 在 {probe_label} 出現 {len(hits)} 次" in wrong_label,
              f"指出號碼其實掛在 {probe_label}:{wrong_label!r}")
        check((window.index, canvas.selected_index) == stay, "提示歸提示,位置仍不動")
        window.find_kind.setCurrentIndex(window.find_kind.findData(probe_label))

        window.find_edit.clear()
        window._find_next(1)
        app.processEvents()
        check((window.index, canvas.selected_index) == stay, "沒填號碼時位置不動")
        check("填一個 track_id" in window.statusBar().currentMessage(),
              f"且說明少了哪一步:{window.statusBar().currentMessage()!r}")

        check(not any(f.dirty for f in window.frames), "整段搜尋沒有改動任何資料")

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
