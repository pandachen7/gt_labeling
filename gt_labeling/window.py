"""主視窗:接線 canvas / 面板 / 清單,負責導覽、存檔與 undo/redo。"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction, QIntValidator, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .dataset import FrameEntry, ImageStore, load_all, scan_root
from .model import Det, FrameLabel, UndoStack, interpolate_missing, load_frame
from .panel import (
    DEFAULT_MAX_GAP,
    BandPanel,
    DetPanel,
    FrameListPanel,
    InterpolatePanel,
    NewDetPanel,
)

UNDO_LIMIT = 60
PREFETCH_RADIUS = 2
DEFAULT_WIDTH = 1680
DEFAULT_HEIGHT = 940

HELP_TEXT = """\
A / ← 上一張    D / → 下一張
滑鼠滾輪 縮放     中鍵 / 右鍵 / Space+左鍵 平移
左鍵拖空白 新增框(套用右上「新框預設」)
  補錨點前先點該 track 任一框,再選「沿用 #id」
左鍵點框 選取,拖角邊改大小,拖框內移動
Delete 刪除選取框    F 還原檢視
Ctrl+S 存檔   Ctrl+Z 復原   Ctrl+Shift+Z 重做
Ctrl+I 補框(選取框的 track)   Ctrl+Shift+I 復原補框"""


class MainWindow(QMainWindow):
    def __init__(self, cache_size: int = 8, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GT 標註修正工具")

        self.settings = QSettings()
        self.entries: list[FrameEntry] = []
        self.frames: list[FrameLabel] = []
        self.undo_stacks: list[UndoStack] = []
        self.index = -1
        self._placement_checked = False
        # 上一次補框前的快照:frame index -> 該幀原本的 dets,用於一鍵整組還原。
        self._last_interp: dict[int, list[Det]] = {}

        self.store = ImageStore(cache_size, self)
        self.store.ready.connect(self._on_image_ready)

        self._build_ui()
        self._build_actions()
        self._restore_settings()
        self._refresh_actions()
        self._refresh_status()

    # ------------------------------------------------------------------ 介面組裝

    def _build_ui(self) -> None:
        self.canvas = ImageCanvas(self)
        self.canvas.new_det_factory = self._make_new_det
        self.list_panel = FrameListPanel(self)
        self.new_panel = NewDetPanel(self)
        self.det_panel = DetPanel(self)
        self.interp_panel = InterpolatePanel(self)
        self.band_panel = BandPanel(self)

        help_label = QLabel(HELP_TEXT, self)
        help_label.setStyleSheet("color: #909090;")

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.new_panel)
        right_layout.addWidget(self.det_panel)
        right_layout.addWidget(self.interp_panel)
        right_layout.addWidget(self.band_panel)
        right_layout.addStretch(1)
        right_layout.addWidget(help_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.list_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 1100, 280])
        self.setCentralWidget(splitter)

        self.lbl_frame = QLabel("—")
        self.lbl_boxes = QLabel("")
        self.lbl_state = QLabel("")
        self.lbl_zoom = QLabel("")
        self.lbl_pos = QLabel("")
        bar = self.statusBar()
        for widget in (self.lbl_frame, self.lbl_boxes, self.lbl_state, self.lbl_zoom):
            bar.addWidget(widget)
        bar.addPermanentWidget(self.lbl_pos)

        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.detsEdited.connect(self._on_dets_edited)
        self.canvas.viewChanged.connect(self._refresh_status)
        self.canvas.navigateRequested.connect(self._navigate)
        self.canvas.hoverMoved.connect(self._on_hover)
        self.list_panel.rowSelected.connect(self._goto)
        self.new_panel.changed.connect(self._on_new_default_changed)
        self.interp_panel.toggled.connect(lambda _: self._refresh_actions())
        self.det_panel.fieldChanged.connect(self._on_field_changed)
        self.band_panel.bandChanged.connect(self._on_band_changed)

    def _build_actions(self) -> None:
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)

        self.act_open = QAction("開資料夾", self)
        self.act_open.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        self.act_open.triggered.connect(self._choose_root)

        self.act_prev = QAction("上一張", self)
        self.act_prev.setShortcut(QKeySequence(Qt.Key.Key_PageUp))
        self.act_prev.triggered.connect(lambda: self._navigate(-1))

        self.act_next = QAction("下一張", self)
        self.act_next.setShortcut(QKeySequence(Qt.Key.Key_PageDown))
        self.act_next.triggered.connect(lambda: self._navigate(1))

        self.act_save = QAction("存檔", self)
        self.act_save.setShortcut(QKeySequence(QKeySequence.StandardKey.Save))
        self.act_save.triggered.connect(self._save_current)

        self.act_undo = QAction("復原", self)
        self.act_undo.setShortcut(QKeySequence(QKeySequence.StandardKey.Undo))
        self.act_undo.triggered.connect(self._undo)

        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcuts(
            [QKeySequence(QKeySequence.StandardKey.Redo), QKeySequence("Ctrl+Y")]
        )
        self.act_redo.triggered.connect(self._redo)

        self.act_delete = QAction("刪除框", self)
        self.act_delete.triggered.connect(self.canvas.delete_selected)

        self.act_fit = QAction("還原檢視", self)
        self.act_fit.triggered.connect(self.canvas.fit_view)

        self.act_autosave = QAction("切幀自動存", self)
        self.act_autosave.setCheckable(True)
        self.act_autosave.setChecked(True)

        self.act_interp = QAction("補框", self)
        self.act_interp.setShortcut(QKeySequence("Ctrl+I"))
        self.act_interp.setToolTip("把選取框所屬 track 的錨點之間補滿(Ctrl+I)")
        self.act_interp.triggered.connect(self._interpolate)

        self.act_interp_undo = QAction("復原補框", self)
        self.act_interp_undo.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.act_interp_undo.triggered.connect(self._undo_interpolation)

        for action in (
            self.act_open, self.act_prev, self.act_next, self.act_save,
            self.act_undo, self.act_redo, self.act_delete, self.act_fit,
            self.act_interp, self.act_interp_undo, self.act_autosave,
        ):
            self.addAction(action)
            toolbar.addAction(action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" 跳到 seq "))
        self.jump_edit = QLineEdit(self)
        self.jump_edit.setFixedWidth(90)
        self.jump_edit.setValidator(QIntValidator(0, 2_000_000_000, self))
        self.jump_edit.returnPressed.connect(self._jump_to_seq)
        toolbar.addWidget(self.jump_edit)
        act_jump = QAction("跳", self)
        act_jump.triggered.connect(self._jump_to_seq)
        toolbar.addAction(act_jump)

    # --------------------------------------------------------------- 設定持久化

    def _restore_settings(self) -> None:
        self.settings.remove("geometry")  # 舊版存過含位置的 geometry,清掉
        self._restore_size()
        self.act_autosave.setChecked(self.settings.value("autosave", True, type=bool))
        self.new_panel.set_label(str(self.settings.value("new_label", "person")))
        self.interp_panel.set_state(
            self.settings.value("interp_on", False, type=bool),
            self.settings.value("interp_gap", DEFAULT_MAX_GAP, type=int),
        )
        self.band_panel.set_band(
            self.settings.value("band_lo", 0.0, type=float),
            self.settings.value("band_hi", 1.0, type=float),
            self.settings.value("band_on", True, type=bool),
        )

    def _restore_size(self) -> None:
        """只還原大小、**不指定位置**。

        記住的座標會在換螢幕、改解析度或高 DPI 縮放後落到可見範圍外,標題列抓不到就搬不動;
        尺寸也必須夾到目前螢幕的可用範圍,否則視窗比桌面大同樣會把標題列頂出去。
        位置交給視窗管理員決定。
        """
        screen = self.screen()
        available = screen.availableGeometry() if screen is not None else None
        max_w = available.width() - 60 if available is not None else DEFAULT_WIDTH
        max_h = available.height() - 80 if available is not None else DEFAULT_HEIGHT
        width = self.settings.value("win_w", DEFAULT_WIDTH, type=int)
        height = self.settings.value("win_h", DEFAULT_HEIGHT, type=int)
        self.resize(max(640, min(width, max_w)), max(480, min(height, max_h)))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._placement_checked:
            self._placement_checked = True
            self._rescue_offscreen()

    def _rescue_offscreen(self) -> None:
        """不指定初始位置,但保證標題列抓得到。

        雙螢幕 + DPI 縮放時,Qt 的邏輯座標可能出現不屬於任何螢幕的空隙
        (例:螢幕 A = 0~1536、螢幕 B 從 1920 起,1536~1920 是空的)。
        視窗管理員的預設位置若落在那段或負座標,左上角就出界、拖不動。
        只有真的落在所有螢幕可用範圍外才搬,使用者自己拖出去的不管。
        """
        frame = self.frameGeometry()
        screens = QApplication.screens()
        if any(s.availableGeometry().contains(frame.topLeft()) for s in screens):
            return

        target = self.screen() or QApplication.primaryScreen()
        if target is None:
            return
        available = target.availableGeometry()
        x = max(available.x(), min(frame.x(), available.right() - frame.width() + 1))
        y = max(available.y(), min(frame.y(), available.bottom() - frame.height() + 1))
        # 用相對位移而非絕對座標:QWidget.pos() 對頂層視窗含邊框、QMoveEvent.pos() 不含,
        # 只搬「差值」就不必依賴哪一種語意。
        self.move(self.pos().x() + (x - frame.x()), self.pos().y() + (y - frame.y()))

    def _save_settings(self) -> None:
        self.settings.setValue("win_w", self.width())
        self.settings.setValue("win_h", self.height())
        self.settings.setValue("autosave", self.act_autosave.isChecked())
        self.settings.setValue("new_label", self.new_panel.label())
        self.settings.setValue("interp_on", self.interp_panel.is_enabled())
        self.settings.setValue("interp_gap", self.interp_panel.max_gap())
        self.settings.setValue("band_lo", self.band_panel.lo_spin.value())
        self.settings.setValue("band_hi", self.band_panel.hi_spin.value())
        self.settings.setValue("band_on", self.band_panel.enabled_box.isChecked())

    def set_band(self, lo: float, hi: float) -> None:
        self.band_panel.set_band(lo, hi, True)

    def last_root(self) -> str | None:
        value = self.settings.value("last_root")
        return str(value) if value else None

    # ------------------------------------------------------------------- 開資料夾

    def _choose_root(self) -> None:
        start = self.last_root() or ""
        chosen = QFileDialog.getExistingDirectory(self, "選擇含 frames/ 與 labels/ 的資料夾", start)
        if chosen:
            self.open_root(Path(chosen))

    def open_root(self, root: Path) -> bool:
        try:
            entries = scan_root(Path(root))
        except (FileNotFoundError, OSError) as exc:
            QMessageBox.critical(self, "開啟失敗", str(exc))
            return False
        if not entries:
            QMessageBox.warning(self, "沒有資料", f"{root}\\labels 下找不到任何 .json")
            return False
        if self.frames and not self._flush_all("切換資料夾"):
            return False

        try:
            frames = load_all(entries)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "讀取標註失敗", str(exc))
            return False

        self.entries = entries
        self.frames = frames
        # 快照裡存的是舊資料集的 frame index,換資料夾後必須丟掉。
        self._last_interp = {}
        self.undo_stacks = []
        for frame in frames:
            stack = UndoStack(UNDO_LIMIT)
            stack.reset(frame.snapshot())
            self.undo_stacks.append(stack)

        self.store.clear()
        self.list_panel.set_frames(frames)
        self.index = -1
        self._goto(0, force=True)
        missing = sum(1 for e in entries if e.image_path is None)
        title = f"GT 標註修正工具 — {root}  ({len(entries)} 幀)"
        self.setWindowTitle(title + (f"  影像缺 {missing}" if missing else ""))
        self.settings.setValue("last_root", str(root))
        return True

    # --------------------------------------------------------------------- 導覽

    def _goto(self, index: int, force: bool = False) -> None:
        if not self.frames:
            return
        index = min(max(index, 0), len(self.frames) - 1)
        if index == self.index and not force:
            return
        if self.index >= 0 and not self._leave_frame(self.index):
            self.list_panel.set_current(self.index)
            return

        self.index = index
        entry = self.entries[index]
        self.canvas.set_frame(self.frames[index], self.store.load_now(entry))
        self.list_panel.set_current(index)
        self._prefetch_around(index)
        self._refresh_actions()
        self._refresh_status()
        self.canvas.setFocus()

    def _navigate(self, delta: int) -> None:
        if self.frames:
            self._goto(self.index + delta)

    def _jump_to_seq(self) -> None:
        text = self.jump_edit.text().strip()
        if not text or not self.frames:
            return
        target = int(text)
        match = next((i for i, f in enumerate(self.frames) if f.seq == target), None)
        if match is None:
            # 沒有完全相符就跳到最接近的 seq。
            match = min(range(len(self.frames)), key=lambda i: abs(self.frames[i].seq - target))
            self.statusBar().showMessage(
                f"沒有 seq={target},跳到最接近的 {self.frames[match].seq}", 4000
            )
        self._goto(match)

    def _prefetch_around(self, index: int) -> None:
        lo = max(0, index - PREFETCH_RADIUS)
        hi = min(len(self.entries), index + PREFETCH_RADIUS + 1)
        self.store.prefetch([self.entries[i] for i in range(lo, hi) if i != index])

    def _on_image_ready(self, seq: int) -> None:
        if 0 <= self.index < len(self.entries) and self.entries[self.index].seq == seq:
            self.canvas.set_pixmap(self.store.peek(seq))

    # --------------------------------------------------------------------- 存檔

    def _leave_frame(self, index: int) -> bool:
        """離開該幀前處理未存修改,回傳是否允許離開。"""
        frame = self.frames[index]
        if not frame.dirty:
            return True
        if self.act_autosave.isChecked():
            return self._write(index)

        answer = QMessageBox.question(
            self,
            "尚未存檔",
            f"seq={frame.seq} 有未存的修改,要存嗎?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._write(index)
        if answer == QMessageBox.StandardButton.Discard:
            self._reload_frame(index)
            return True
        return False

    def _write(self, index: int) -> bool:
        frame = self.frames[index]
        try:
            frame.save()
        except OSError as exc:
            QMessageBox.critical(self, "存檔失敗", f"{frame.path}\n{exc}")
            return False
        self.list_panel.refresh_row(index, frame)
        self.list_panel.refresh_summary(self.frames)
        self._refresh_status()
        return True

    def _save_current(self) -> None:
        if 0 <= self.index < len(self.frames) and self._write(self.index):
            self.statusBar().showMessage(f"已存 {self.frames[self.index].path.name}", 2500)

    def _reload_frame(self, index: int) -> None:
        frame = load_frame(self.entries[index].label_path)
        self.frames[index] = frame
        stack = UndoStack(UNDO_LIMIT)
        stack.reset(frame.snapshot())
        self.undo_stacks[index] = stack
        self.list_panel.refresh_row(index, frame)
        self.list_panel.refresh_summary(self.frames)

    def _flush_all(self, reason: str) -> bool:
        dirty = [i for i, f in enumerate(self.frames) if f.dirty]
        if not dirty:
            return True
        if self.act_autosave.isChecked():
            return all(self._write(i) for i in dirty)

        answer = QMessageBox.question(
            self,
            reason,
            f"還有 {len(dirty)} 幀未存,要全部存起來嗎?",
            QMessageBox.StandardButton.SaveAll
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.SaveAll,
        )
        if answer == QMessageBox.StandardButton.SaveAll:
            return all(self._write(i) for i in dirty)
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if self.frames and not self._flush_all("關閉程式"):
            event.ignore()
            return
        self._save_settings()
        super().closeEvent(event)

    # ----------------------------------------------------------------- 新框預設

    def _next_track_id(self) -> int:
        """新框要掛的 track_id。

        選了「沿用」就直接用記住的號碼(補錨點時連畫好幾幀都屬於同一條軌跡);
        否則取當前幀最大 +1——用途是補該幀漏標的框,號碼接在**這一幀**已有的
        最後一個之後。取 max 而非 dets 陣列最後一顆:tracker 輸出的順序不保證
        按 id 遞增(實測有 [4,2,1,3,5]),取最後一顆遇到亂序就會發出已佔用的號。
        """
        follow = self.new_panel.follow_id()
        if follow is not None:
            return follow
        if not (0 <= self.index < len(self.frames)):
            return 0
        used = [d.track_id for d in self.frames[self.index].dets if d.track_id is not None]
        return max(used) + 1 if used else 0

    def _make_new_det(self) -> Det:
        label, ppe = self.new_panel.defaults()
        return Det(label=label, track_id=self._next_track_id(), ppe=ppe)

    def _on_new_default_changed(self, label: str) -> None:
        self.statusBar().showMessage(f"新框預設:{label}", 2000)
        # 焦點還給畫布,切完模式可以直接畫、也不吃掉 A/D 之類的快捷鍵。
        self.canvas.setFocus()

    # ----------------------------------------------------------------- 內插補框

    def _interpolate(self) -> None:
        """前置條件不足時一律說明原因。

        這個動作只有快捷鍵入口的使用者最多,靜默失敗(把 action 設成 disabled)
        會讓人完全不知道自己少做了哪一步。
        """
        if not self.frames:
            return
        if not self.interp_panel.is_enabled():
            self.statusBar().showMessage(
                "補框未啟用:先在右側「內插補框」勾選「啟用內插補框」", 6000)
            return
        det = self.canvas.selected_det
        if det is None:
            self.statusBar().showMessage(
                "先點選一個框:補框要靠它決定補哪一條 track", 6000)
            return
        if det.track_id is None:
            self.statusBar().showMessage(
                "選取的框沒有 track_id,無法決定要補哪一條 track", 6000)
            return

        max_gap = self.interp_panel.max_gap()
        plan = interpolate_missing(self.frames, det.track_id, max_gap)
        if not plan.additions:
            if plan.skipped:
                shortest = min(g for _, _, g in plan.skipped)
                QMessageBox.information(
                    self,
                    "沒有補上任何框",
                    f"track {det.track_id} 有 {len(plan.skipped)} 個洞,"
                    f"但全部超過門檻 {max_gap} 幀(最短的間距 {shortest} 幀)。\n\n"
                    f"補框只在錨點夠密時才準,所以刻意不去猜這種長洞——"
                    f"它也可能是目標被遮擋、本來就不該有框。\n\n"
                    f"做法:先在洞中間每隔 {max_gap} 幀左右手動畫一個框(記得填同一個 "
                    f"track_id {det.track_id}),再按一次補框;\n"
                    f"或把門檻放寬到 {shortest} 幀以上——但誤差會明顯變大,"
                    f"補完要逐幀檢查。",
                )
            else:
                self.statusBar().showMessage(
                    f"track {det.track_id} 每一幀都有框,沒有洞需要補", 5000)
            return

        touched = plan.frame_indexes
        detail = (f"\n\n另有 {len(plan.skipped)} 個洞間距超過 "
                  f"{self.interp_panel.max_gap()} 幀被跳過(可能是遮擋),需要你手動補錨點。"
                  if plan.skipped else "")
        answer = QMessageBox.question(
            self,
            "補框",
            f"track {det.track_id}:要在 {len(touched)} 幀補上 {len(plan.additions)} 個框嗎?"
            f"{detail}",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if answer == QMessageBox.StandardButton.Ok:
            self.apply_interpolation(plan)

    def apply_interpolation(self, plan) -> None:
        """套用一份補框計畫。與 ``_interpolate`` 的確認對話框分開,方便直接驗收。"""
        touched = plan.frame_indexes
        if not touched:
            return
        self._last_interp = {i: self.frames[i].snapshot() for i in touched}
        for index, new_det in plan.additions:
            self.frames[index].dets.append(new_det)
        for index in touched:
            self.undo_stacks[index].commit(self.frames[index].snapshot())

        self._after_bulk_change(touched)
        seqs = [self.frames[i].seq for i in touched]
        self.statusBar().showMessage(
            f"已補 {len(plan.additions)} 框(seq {min(seqs)}..{max(seqs)}),"
            f"Ctrl+Shift+I 可整組復原",
            6000,
        )

    def _undo_interpolation(self) -> None:
        if not self._last_interp:
            self.statusBar().showMessage("沒有可復原的補框", 3000)
            return
        touched = sorted(self._last_interp)
        for index, snapshot in self._last_interp.items():
            self.frames[index].restore(snapshot)
            self.undo_stacks[index].commit(self.frames[index].snapshot())
        self._last_interp = {}
        self._after_bulk_change(touched)
        self.statusBar().showMessage(f"已復原 {len(touched)} 幀的補框", 4000)

    def _after_bulk_change(self, touched: list[int]) -> None:
        """跨幀改動後的統一刷新。"""
        for index in touched:
            self.list_panel.refresh_row(index, self.frames[index])
        self.list_panel.refresh_summary(self.frames)
        self.canvas.reload_dets()
        if 0 <= self.index < len(self.frames):
            self.det_panel.set_det(self.canvas.selected_det, self.frames[self.index].size)
        self._refresh_actions()
        self._refresh_status()

    # ------------------------------------------------------------- 編輯與 undo

    def _on_selection_changed(self, index: int) -> None:
        frame = self.frames[self.index] if 0 <= self.index < len(self.frames) else None
        selected = self.canvas.selected_det
        self.det_panel.set_det(selected, frame.size if frame is not None else None)
        if selected is not None:
            self.new_panel.set_follow_candidate(selected.track_id)
        # 走統一刷新而非單獨設某個 action:選取狀態會影響不只一個動作,
        # 逐一手動更新遲早漏掉(「補框」就是這樣一直停在停用狀態)。
        self._refresh_actions()

    def _on_dets_edited(self) -> None:
        if not (0 <= self.index < len(self.frames)):
            return
        self.undo_stacks[self.index].commit(self.frames[self.index].snapshot())
        self._after_model_change()

    def _on_field_changed(self, field: str, value) -> None:
        det = self.canvas.selected_det
        if det is None:
            return
        if field == "label":
            if det.label == value:
                return
            det.label = value
            if not det.is_person:
                det.ppe = None
        elif field == "track_id":
            if det.track_id == value:
                return
            det.track_id = value
        elif field == "ppe":
            if det.ppe == value:
                return
            det.ppe = value
        else:
            return
        self.canvas.update()
        self._on_dets_edited()

    def _after_model_change(self) -> None:
        frame = self.frames[self.index]
        self.list_panel.refresh_row(self.index, frame)
        self.list_panel.refresh_summary(self.frames)
        self.det_panel.set_det(self.canvas.selected_det, frame.size)
        self._refresh_actions()
        self._refresh_status()

    def _undo(self) -> None:
        self._apply_history(redo=False)

    def _redo(self) -> None:
        self._apply_history(redo=True)

    def _apply_history(self, redo: bool) -> None:
        if not (0 <= self.index < len(self.frames)):
            return
        stack = self.undo_stacks[self.index]
        snapshot = stack.redo() if redo else stack.undo()
        if snapshot is None:
            return
        self.frames[self.index].restore(snapshot)
        self.canvas.reload_dets()
        self._after_model_change()

    def _on_band_changed(self, lo, hi) -> None:
        self.canvas.set_band(lo, hi)

    def _on_hover(self, nx: float, ny: float) -> None:
        if not (0 <= self.index < len(self.frames)):
            return
        width, height = self.frames[self.index].size
        self.lbl_pos.setText(
            f"n=({nx:.5f}, {ny:.5f})  px=({nx * width:.0f}, {ny * height:.0f})"
        )

    # ------------------------------------------------------------------- 狀態列

    def _refresh_actions(self) -> None:
        has_data = bool(self.frames)
        stack = self.undo_stacks[self.index] if 0 <= self.index < len(self.frames) else None
        self.act_prev.setEnabled(has_data and self.index > 0)
        self.act_next.setEnabled(has_data and self.index < len(self.frames) - 1)
        self.act_save.setEnabled(has_data)
        self.act_fit.setEnabled(has_data)
        self.act_undo.setEnabled(bool(stack and stack.can_undo))
        self.act_redo.setEnabled(bool(stack and stack.can_redo))
        self.act_delete.setEnabled(self.canvas.selected_det is not None)
        # 刻意不依「已勾選 / 已選框」停用:停用的動作按下去毫無反應,
        # 使用者無從得知少了哪一步。改由 _interpolate 逐項說明原因。
        self.act_interp.setEnabled(has_data)
        self.act_interp_undo.setEnabled(bool(self._last_interp))

    def _refresh_status(self) -> None:
        if not (0 <= self.index < len(self.frames)):
            self.lbl_frame.setText("未開啟資料")
            for label in (self.lbl_boxes, self.lbl_state, self.lbl_zoom, self.lbl_pos):
                label.setText("")
            return
        frame = self.frames[self.index]
        self.lbl_frame.setText(f"{self.index + 1}/{len(self.frames)}   seq={frame.seq}")
        self.lbl_boxes.setText(f"  框 {len(frame.dets)}  待補 {frame.pending_count}")
        self.lbl_state.setText("  未存 *" if frame.dirty else "  已存")
        self.lbl_zoom.setText(f"  zoom {self.canvas.tf.zoom * 100:.0f}%")
