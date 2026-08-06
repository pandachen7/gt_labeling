"""主視窗:接線 canvas / 面板 / 清單,負責導覽、存檔與 undo/redo。"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction, QIntValidator, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .dataset import FrameEntry, ImageStore, load_all, scan_root
from .model import (
    LABELS,
    Det,
    FrameLabel,
    Remap,
    TrackDelete,
    UndoStack,
    find_track,
    interpolate_missing,
    load_frame,
    plan_remap,
    plan_track_delete,
)
from .panel import (
    DEFAULT_MAX_GAP,
    BandPanel,
    DeleteTrackDialog,
    DetPanel,
    FrameListPanel,
    InterpolatePanel,
    NewDetPanel,
    RemapRangeDialog,
)

UNDO_LIMIT = 60
PREFETCH_RADIUS = 2
DEFAULT_WIDTH = 1680
DEFAULT_HEIGHT = 940
# 撞號警告最多列幾個 seq:列完 600 幀會把對話框撐出螢幕,反而看不到按鈕。
CONFLICT_PREVIEW = 12

HELP_TEXT = """\
A / ← 上一張    D / → 下一張    Home / End 首幀 / 末幀
滑鼠滾輪 縮放     中鍵 / 右鍵 / Space+左鍵 平移
左鍵拖空白 新增框(套用右上「新框預設」)
  預設「沿用 #id」:先點該 track 任一框,之後畫的都掛同一號
左鍵點框 選取,拖角邊改大小,拖框內移動
Delete 刪除選取框(焦點在畫布)    F 還原檢視
Ctrl+S 存檔   Ctrl+Z 復原   Ctrl+Shift+Z 重做
Ctrl+I 補框(選取框的 track)   Ctrl+R 改 id(整條軌跡換號)
幀清單 Shift / Ctrl 多選 → 刪整段軌跡(Delete / Ctrl+Shift+D)
  或只改那一段的 id(Ctrl+Shift+R),範圍外維持舊號
Ctrl+Shift+I 復原上次批次(補框 / 改 id / 刪軌跡)
Ctrl+F 找 track(F3 下一個 / Shift+F3 上一個)
  搜尋欄內:Enter 下一個 / Shift+Enter 上一個 / Esc 回畫布"""


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
        # 上一次跨幀批次改動前的快照:frame index -> 該幀原本的 dets,用於一鍵整組還原。
        # 補框與改 id 共用一份:兩者都是「一次動很多幀」,單幀 Ctrl+Z 救不回來。代價是
        # 後做的那次會蓋掉前一次的還原點,所以訊息一律講清楚復原的是哪一次操作。
        self._last_bulk: dict[int, list[Det]] = {}
        self._last_bulk_what = ""
        # 最後點過的框屬於哪一條軌跡。只當「刪除軌跡」對話框的預填值——真正送出去
        # 的是對話框裡那一組,所以這份記憶不準也不會默默刪錯東西。
        self._last_track: tuple[str, int] | None = None
        # 刪除進行中,黏著記憶暫停更新。見 ``_frozen_track_memory``。
        self._freeze_track_memory = False

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

        # 右欄包進捲動區:不包的話,這一疊面板的最小高度會直接變成**視窗**的最小
        # 高度(實測 767px,再加選單列與狀態列就要 836px),而 QMainWindow 一旦發現
        # 視窗比最小高度小就會自己把視窗頂大、而且只長不縮。後果是選一個框
        # (det_panel 要多顯示三行 bbox 座標)視窗就莫名其妙長高一截,在 150% DPI
        # 縮放的螢幕上(邏輯高度只剩 720)底部整個被推出畫面外。
        # 包起來之後這一疊要多高都行,塞不下就捲動,視窗尺寸由使用者說了算。
        right_scroll = QScrollArea(self)
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.list_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_scroll)
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
        self.canvas.deleteRequested.connect(self._delete_selected)
        self.canvas.hoverMoved.connect(self._on_hover)
        self.list_panel.rowSelected.connect(self._goto)
        self.list_panel.deleteRequested.connect(self._delete_track_range)
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

        self.act_first = QAction("首幀", self)
        self.act_first.setShortcut(QKeySequence(Qt.Key.Key_Home))
        self.act_first.triggered.connect(lambda: self._goto(0))

        self.act_last = QAction("末幀", self)
        self.act_last.setShortcut(QKeySequence(Qt.Key.Key_End))
        self.act_last.triggered.connect(lambda: self._goto(len(self.frames) - 1))

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
        self.act_delete.triggered.connect(self._delete_selected)

        self.act_remap_range = QAction("改選取幀內的軌跡 id…", self)
        self.act_remap_range.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.act_remap_range.setToolTip(
            "只把選取的那幾幀裡的 track 換號,範圍外維持舊號 —— 用在 tracker 把"
            "某一段誤配給別的目標時(Ctrl+Shift+R)")
        self.act_remap_range.triggered.connect(self._remap_range)

        self.act_delete_track = QAction("刪除選取幀內的軌跡…", self)
        self.act_delete_track.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self.act_delete_track.setToolTip(
            "跳出對話框,把指定的 track 在幀清單選取的那幾幀裡整段刪掉"
            "(Ctrl+Shift+D,或在幀清單上按 Delete)")
        self.act_delete_track.triggered.connect(self._delete_track_range)

        self.act_fit = QAction("還原檢視", self)
        self.act_fit.triggered.connect(self.canvas.fit_view)

        self.act_autosave = QAction("切幀自動存", self)
        self.act_autosave.setCheckable(True)
        self.act_autosave.setChecked(True)

        self.act_interp = QAction("補框", self)
        self.act_interp.setShortcut(QKeySequence("Ctrl+I"))
        self.act_interp.setToolTip("把選取框所屬 track 的錨點之間補滿(Ctrl+I)")
        self.act_interp.triggered.connect(self._interpolate)

        self.act_remap = QAction("改 id", self)
        self.act_remap.setShortcut(QKeySequence("Ctrl+R"))
        self.act_remap.setToolTip(
            "把選取框所屬 track 的 id 在所有幀改成指定號碼,用來接回斷軌(Ctrl+R)")
        self.act_remap.triggered.connect(self._remap_track)

        self.act_bulk_undo = QAction("復原上次批次", self)
        self.act_bulk_undo.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.act_bulk_undo.setToolTip(
            "整組還原上一次跨幀批次改動(補框 / 改 id)(Ctrl+Shift+I)")
        self.act_bulk_undo.triggered.connect(self._undo_last_bulk)

        self.act_find = QAction("找 track", self)
        self.act_find.setShortcut(QKeySequence(QKeySequence.StandardKey.Find))
        self.act_find.setToolTip(
            "聚焦工具列的 track 搜尋欄,有選取框就預填它的軌跡(Ctrl+F)")
        self.act_find.triggered.connect(self._focus_find)

        self.act_find_next = QAction("找", self)
        self.act_find_next.setShortcut(QKeySequence(Qt.Key.Key_F3))
        self.act_find_next.setToolTip(
            "跳到下一個出現該 track 的框(F3,或在搜尋欄按 Enter)")
        self.act_find_next.triggered.connect(lambda: self._find_next(1))

        self.act_find_prev = QAction("找上一個", self)
        self.act_find_prev.setShortcut(QKeySequence("Shift+F3"))
        self.act_find_prev.setToolTip(
            "跳到上一個出現該 track 的框(Shift+F3,或在搜尋欄按 Shift+Enter)")
        self.act_find_prev.triggered.connect(lambda: self._find_next(-1))

        menu_file = self.menuBar().addMenu("檔案")
        menu_file.addAction(self.act_open)
        menu_file.addAction(self.act_save)

        menu_edit = self.menuBar().addMenu("編輯")
        menu_edit.addAction(self.act_undo)
        menu_edit.addAction(self.act_redo)
        menu_edit.addSeparator()
        menu_edit.addAction(self.act_delete)
        menu_edit.addAction(self.act_delete_track)
        menu_edit.addAction(self.act_remap_range)
        menu_edit.addAction(self.act_interp)
        menu_edit.addAction(self.act_remap)
        menu_edit.addAction(self.act_bulk_undo)

        menu_view = self.menuBar().addMenu("檢視")
        menu_view.addAction(self.act_fit)

        # 進了選單還是要 addAction:快捷鍵綁在 QAction 本身,不靠選單成立,
        # 而導覽動作(上一張 / 下一張)兩處都不擺,少了這裡就整組失效。
        # 它們不需要按鈕——入口是鍵盤與畫布,擺出來只是佔位。
        for action in (
            self.act_open, self.act_save, self.act_undo, self.act_redo,
            self.act_delete, self.act_delete_track, self.act_remap_range,
            self.act_fit, self.act_interp, self.act_remap, self.act_bulk_undo,
            self.act_autosave, self.act_prev, self.act_next,
        ):
            self.addAction(action)

        toolbar.addAction(self.act_autosave)

        # Home / End 綁在畫布而非整個視窗:視窗層的快捷鍵會連 jump_edit 打字時
        # 一起攔掉,那裡的 Home / End 必須留給行首 / 行尾。幀清單有焦點時
        # QListWidget 自己跳第一 / 最後一列,經 rowSelected 走到同一個 _goto。
        for action in (self.act_first, self.act_last):
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            self.canvas.addAction(action)

        # 三個都得 addAction 快捷鍵才生效,但只有「找」會進 toolbar:搜尋欄本體就在
        # 下面,再擺兩顆按鈕會把工具列塞爆。Ctrl+F / Shift+F3 的入口寫在右下角說明。
        for action in (self.act_find, self.act_find_next, self.act_find_prev):
            self.addAction(action)

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

        self._build_find_box(toolbar)

    def _build_find_box(self, toolbar: QToolBar) -> None:
        """工具列上的 track 搜尋欄:label 下拉 + 號碼 + 「找」。

        label 下拉排在號碼前面,讀起來就是 ``person #7``——軌跡身分是
        ``(label, track_id)``,把 label 擺在明處才不會讓人以為號碼是全域唯一的。
        """
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" 找 track "))

        self.find_kind = QComboBox(self)
        self.find_kind.addItem("全部", None)
        for name in LABELS:
            self.find_kind.addItem(name, name)
        self.find_kind.setFixedWidth(88)
        self.find_kind.setToolTip(
            "限定 label。軌跡身分是 (label, track_id):person #7 與 drone #7 是"
            "兩條不同軌跡,選「全部」會兩條都停")
        toolbar.addWidget(self.find_kind)

        self.find_edit = QLineEdit(self)
        self.find_edit.setFixedWidth(80)
        self.find_edit.setPlaceholderText("track_id")
        self.find_edit.setValidator(QIntValidator(0, 2_000_000_000, self))
        self.find_edit.returnPressed.connect(lambda: self._find_next(1))
        toolbar.addWidget(self.find_edit)
        toolbar.addAction(self.act_find_next)

        # Shift+Enter 與 Esc 綁在搜尋欄本身(WidgetShortcut)。Shift+Enter 非走
        # shortcut 不可:QLineEdit 收到 Return 一律發 returnPressed、不分 modifier,
        # 不先攔下來就會被當成「找下一個」。數字鍵盤的 Enter 是另一個 key,一併綁。
        back = QAction("上一個", self.find_edit)
        back.setShortcuts([QKeySequence("Shift+Return"), QKeySequence("Shift+Enter")])
        back.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        back.triggered.connect(lambda: self._find_next(-1))
        self.find_edit.addAction(back)

        leave = QAction("離開搜尋欄", self.find_edit)
        leave.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        leave.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        leave.triggered.connect(lambda: self.canvas.setFocus())
        self.find_edit.addAction(leave)

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
        self._last_bulk = {}
        self._last_bulk_what = ""
        # 記住的軌跡同理:新資料集的同一個號碼是完全不同的目標,留著只會誤導。
        self._last_track = None
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
        # 焦點還給畫布,讓點完清單 / 跳完 seq 就能直接用 A、D 翻頁——但人正在幀清單
        # 裡操作時不能搶:選一列就切一次幀,搶走焦點會讓接下來的 ↑↓ 與 Delete
        # (刪選取幀內的整條軌跡)全部落空,而拉範圍本來就是連續好幾次選取。
        if not self.list_panel.list.hasFocus():
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

    # ----------------------------------------------------------------- 找 track

    def _focus_find(self) -> None:
        """Ctrl+F:聚焦搜尋欄,有選取框就把它的整組 ``(label, id)`` 預填進去。

        連 label 一起預填而不只填號碼:這個功能最常見的用法是「追我正在看的這條
        軌跡」,而軌跡身分是 ``(label, track_id)``——只填號碼的話,person #7 與
        drone #7 會混在一起輪流跳,追軌變成在兩條線之間來回。
        """
        det = self.canvas.selected_det
        if det is not None and det.track_id is not None:
            self.find_edit.setText(str(det.track_id))
            index = self.find_kind.findData(det.label)
            if index >= 0:
                self.find_kind.setCurrentIndex(index)
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _find_next(self, step: int) -> None:
        """跳到下一個(``step=1``)/ 上一個(``step=-1``)出現搜尋目標的框並選取它。

        位置用 ``(frame index, det index)`` 字典序比較,所以同一幀有兩個同號框時
        會逐個停下來,不會整幀跳過——撞號(清單上的 DUP)要看的正是第二個框。
        起點含「目前選取的框」,沒選框時當前幀的第一個命中會先被選起來,不會直接
        跳走:人多半是先想確認「這一幀到底有沒有」。

        掃到盡頭繞回另一端並在訊息註明。不繞的話追到最後一幀就得手動回頭,而
        「這條軌跡總共出現幾次、現在是第幾次」正是最想順便知道的事。

        前置不足時一律說明原因(同 ``_interpolate`` / ``_remap_track``):這功能
        多半靠快捷鍵觸發,靜默失敗會讓人不知道少做了哪一步。
        """
        if not self.frames:
            return
        text = self.find_edit.text().strip()
        if not text:
            self.statusBar().showMessage(
                "先在工具列「找 track」填一個 track_id(Ctrl+F 直接聚焦)", 6000)
            self.find_edit.setFocus()
            return

        track_id = int(text)
        label = self.find_kind.currentData()
        hits = find_track(self.frames, label, track_id)
        what = f"{label} #{track_id}" if label else f"#{track_id}(不分 label)"
        if not hits:
            self.statusBar().showMessage(f"整份資料集找不到 {what}", 5000)
            self._warn_not_found(label, track_id, what)
            return

        here = (self.index, self.canvas.selected_index)
        if step > 0:
            pos = next((n for n, hit in enumerate(hits) if hit > here), None)
        else:
            pos = next((n for n in reversed(range(len(hits))) if hits[n] < here), None)
        wrapped = pos is None
        if pos is None:
            pos = 0 if step > 0 else len(hits) - 1

        # 焦點原樣還回去:_goto 尾端會把焦點搶到畫布,但從搜尋欄按 Enter 的人要能
        # 連按下去,從畫布按 F3 的人也不該被搬進輸入框(那會讓 A/D 翻幀失效)。
        focused = QApplication.focusWidget()
        frame_index, det_index = hits[pos]
        self._goto(frame_index)
        if self.index != frame_index:  # 存檔對話框被取消,留在原地
            return
        self.canvas.select(det_index)
        if focused is not None:
            focused.setFocus()

        if len(hits) == 1:
            note = "(整份資料集只出現這一次)"
        elif wrapped:
            note = "(已繞回開頭)" if step > 0 else "(已繞回結尾)"
        else:
            note = ""
        self.statusBar().showMessage(
            f"{what}  第 {pos + 1}/{len(hits)} 次出現  seq={self.frames[frame_index].seq}"
            f"{note}",
            6000,
        )

    def _warn_not_found(self, label: str | None, track_id: int, what: str) -> None:
        """找不到就擋下來說清楚,不只在狀態列閃一下。

        找不到多半不是「這條軌跡真的不存在」,而是**下拉的 label 選錯了**——
        軌跡身分是 ``(label, track_id)``,而 person 與 drone 常各自從 0 開始編號,
        同一個號碼兩邊都有人用。所以限定 label 落空時再用不分 label 掃一次,把
        「號碼掛在哪」直接寫進對話框;沒有這一句,人只會反覆確認自己有沒有打錯字。
        """
        hint = ""
        if label is not None:
            elsewhere = find_track(self.frames, None, track_id)
            if elsewhere:
                others = sorted({self.frames[i].dets[k].label for i, k in elsewhere})
                hint = (f"\n\n但 #{track_id} 在 {' / '.join(others)} 出現 "
                        f"{len(elsewhere)} 次——label 下拉可能選錯了。")
        QMessageBox.warning(self, "找不到 track", f"整份資料集沒有 {what}。{hint}")

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
        plan = interpolate_missing(self.frames, det.label, det.track_id, max_gap)
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
        self._snapshot_bulk(touched, f"補 {len(plan.additions)} 框")
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

    # -------------------------------------------------------------------- 改 id

    def _remap_track(self) -> None:
        """把選取框所屬 track 的 id 在所有幀換成指定號碼。

        tracker 斷軌時同一個目標會被切成兩個號碼,逐幀改號很慢,所以整條一次換。

        入口與補框一致(先點框再按),前置條件不足時一律說明原因而非停用按鈕:
        這個動作多半靠快捷鍵觸發,靜默失敗會讓人完全不知道少做了哪一步。
        """
        if not self.frames:
            return
        det = self.canvas.selected_det
        if det is None:
            self.statusBar().showMessage(
                "先點選一個框:改 id 要靠它決定改哪一條 track", 6000)
            return
        if det.track_id is None:
            self.statusBar().showMessage(
                "選取的框沒有 track_id,先在右側「選取的框」填一個號碼再改", 6000)
            return

        label, old_id = det.label, det.track_id
        new_id, confirmed = QInputDialog.getInt(
            self,
            "改 id",
            f"把 {label} #{old_id} 在所有幀改成:",
            old_id,
            0,
            2_000_000_000,
            1,
        )
        if not confirmed:
            return
        if new_id == old_id:
            self.statusBar().showMessage("目標與來源相同,沒有改動", 4000)
            return

        plan = plan_remap(self.frames, label, old_id, new_id)
        if not plan.targets:
            self.statusBar().showMessage(f"找不到 {label} #{old_id} 的框", 4000)
            return

        summary = (f"{label} #{old_id} → #{new_id}\n"
                   f"共 {len(plan.frame_indexes)} 幀 / {plan.box_count} 框會改號。")
        if plan.conflicts:
            listed = ", ".join(str(s) for s in plan.conflicts[:CONFLICT_PREVIEW])
            more = (f" …等 {len(plan.conflicts)} 幀"
                    if len(plan.conflicts) > CONFLICT_PREVIEW else "")
            answer = QMessageBox.warning(
                self,
                "改 id — 會撞號",
                f"{summary}\n\n"
                f"但有 {len(plan.conflicts)} 幀已經存在 {label} #{new_id},"
                f"改完那幾幀會同時出現兩個 #{new_id}:\n"
                f"  seq {listed}{more}\n\n"
                f"斷軌的兩段通常各佔不同的幀、不會重疊,所以這多半代表這兩段其實"
                f"不是同一個目標。仍要改嗎?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
        else:
            answer = QMessageBox.question(
                self,
                "改 id",
                f"{summary}\n\n要改嗎?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok,
            )
        if answer == QMessageBox.StandardButton.Ok:
            self.apply_remap(plan, label, old_id, new_id)

    def _remap_preview(
        self, rows: list[int], label: str, old_id: int, new_id: int
    ) -> tuple[int, str]:
        """對話框用的即時試算:這一段改號會動到幾個框、之後會變成什麼樣子。"""
        if old_id == new_id:
            return 0, "新舊號碼相同,沒有東西要改。"
        plan = plan_remap(self.frames, label, old_id, new_id, rows)
        if not plan.targets:
            return 0, f"選取的這幾幀裡沒有 {label} #{old_id} 的框。"

        lines = [f"命中 {len(plan.frame_indexes)} 幀 / {plan.box_count} 個框。"]
        # 改完之後這條軌跡會長什麼樣,是按下確定前最該知道的事:拆成兩段?接上
        # 既有的另一段?人在清單上拉範圍時這兩件事都看不出來。
        lines.append(
            f"範圍外還有 {plan.outside} 個框留著 #{old_id}(這條會拆成兩段)。"
            if plan.outside
            else f"範圍已涵蓋整條軌跡,#{old_id} 會從整份資料集消失。"
        )
        if plan.merges:
            lines.append(f"範圍外已有 {plan.merges} 個框是 #{new_id},改完接成同一條。")
        if plan.conflicts:
            listed = ", ".join(str(s) for s in plan.conflicts[:CONFLICT_PREVIEW])
            more = (f" …等 {len(plan.conflicts)} 幀"
                    if len(plan.conflicts) > CONFLICT_PREVIEW else "")
            lines.append(
                f"⚠ {len(plan.conflicts)} 幀改完會同時出現兩個 #{new_id}:"
                f"seq {listed}{more}")
        return plan.box_count, "\n".join(lines)

    def _remap_range(self) -> None:
        """只把選取的那幾幀裡的軌跡換號,範圍外維持舊號。

        用在 tracker 把某一段誤配給別的目標:只有那一段要拆出來,``Ctrl+R`` 的
        全域換號會把本來正確的部分一起改壞。

        入口與「刪除選取幀內的軌跡」對稱(先圈範圍再開對話框),與 ``Ctrl+R``
        的「先點框」刻意分開:兩種前置條件混在同一個入口只會讓人搞不清該先做哪
        一步。
        """
        if not self.frames:
            return
        rows = self.list_panel.selected_rows()
        if not rows:
            self.statusBar().showMessage(
                "先在左邊幀清單選要改號的那幾幀(Shift 拉連續 / Ctrl 加點零散的)", 6000)
            return

        seqs = [self.frames[i].seq for i in rows]
        dialog = RemapRangeDialog(
            f"選取 {len(rows)} 幀(seq {min(seqs)}..{max(seqs)})",
            lambda label, old_id, new_id: self._remap_preview(rows, label, old_id, new_id),
            self._last_track,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.target()
        if target is None:
            return
        label, old_id, new_id = target
        plan = plan_remap(self.frames, label, old_id, new_id, rows)
        if plan.targets:
            self.apply_remap(plan, label, old_id, new_id)

    def apply_remap(self, plan: Remap, label: str, old_id: int, new_id: int) -> None:
        """套用一份換號計畫。與 ``_remap_track`` 的對話框分開,方便直接驗收。"""
        touched = plan.frame_indexes
        if not touched:
            return
        self._snapshot_bulk(touched, f"{label} #{old_id} → #{new_id}")
        for _, target in plan.targets:
            target.track_id = new_id
        for index in touched:
            self.undo_stacks[index].commit(self.frames[index].snapshot())

        self._after_bulk_change(touched)
        # 「沿用 #id」是黏著值,不會因為框的內容改變而更新;選取框既然換了號,
        # 接著畫的新框要跟著新號碼,否則會默默補出剛剛才淘汰掉的舊號。
        selected = self.canvas.selected_det
        if selected is not None and selected.track_id == new_id:
            self.new_panel.set_follow_candidate(new_id)
        # 記住的軌跡剛被換掉號碼,舊號在資料集裡已經不存在了,跟著改成新號;
        # 不改的話,「刪除軌跡」對話框會預填一個查無此人的號碼。
        if self._last_track == (label, old_id):
            self._last_track = (label, new_id)

        seqs = [self.frames[i].seq for i in touched]
        self.statusBar().showMessage(
            f"已把 {label} #{old_id} 改成 #{new_id}({len(touched)} 幀 / "
            f"{plan.box_count} 框,seq {min(seqs)}..{max(seqs)}),"
            f"Ctrl+Shift+I 可整組復原",
            6000,
        )

    # --------------------------------------------------- 刪掉一條軌跡的某一段

    def _remember_track(self) -> None:
        """把「使用者現在正在處理哪一條軌跡」同步到兩個黏著入口。

        黏著值而非即時讀 ``canvas.selected_det``:切幀時 ``set_frame`` 會清掉畫布
        選取,而這功能的操作順序正是「點框 → 到左邊清單圈一段幀 → Delete」,中間
        必然切過幀,真要現場問就永遠問不到。理由同 ``NewDetPanel`` 的「沿用 #id」。

        兩個入口一起更新而不各自維護:它們問的是同一件事。分開更新就會像先前那樣
        漏掉一半:在右側面板改掉號碼後,刪除目標跟上了,「沿用 #id」卻還停在舊號,
        接著畫的錨點默默掛回剛淘汰掉的號碼。

        沒有 track_id 的框不覆蓋既有值:那種框連自己是哪一條都還沒定,拿它當
        刪除目標或新框號碼都沒有意義,反而會把人剛選好的目標洗掉。

        刪除期間整個停手:記的是「使用者主動指定了哪一條」,而刪完選取會遞補到
        隔壁的框——那是 index 位移的副作用,不是指定。見 ``_frozen_track_memory``。
        """
        if self._freeze_track_memory:
            return
        det = self.canvas.selected_det
        if det is not None and det.track_id is not None:
            self._last_track = (det.label, det.track_id)
            self.new_panel.set_follow_candidate(det.track_id)

    @contextmanager
    def _frozen_track_memory(self) -> Generator[None, None, None]:
        """這段期間的選取變動不算「使用者指定的軌跡」。

        刪除會從兩條路各推一次記憶更新——``selectionChanged`` 的遞補選取,以及
        ``detsEdited`` → ``_after_model_change`` 的收尾——所以擋在 ``_remember_track``
        這個共同出口,而不是各自的訊號處理器。

        用範圍而非黏著旗標:旗標一路留著的話,刪完接著在右側面板改 track_id 就更新
        不到記憶了。凍結只在刪除這一輪內有效。
        """
        self._freeze_track_memory = True
        try:
            yield
        finally:
            self._freeze_track_memory = False

    def _delete_preview(self, rows: list[int], label: str, track_id: int) -> tuple[int, str]:
        """對話框用的即時試算:這組 (label, id) 在選取的那幾幀裡會刪掉幾個框。

        「範圍外還剩幾個」是刪之前最該知道的事:剩 0 代表這條軌跡會整條消失,
        而人在清單上拉範圍時很難確定自己到底圈到哪裡。
        """
        plan = plan_track_delete(self.frames, label, track_id, rows)
        if not plan.targets:
            return 0, f"選取的這幾幀裡沒有 {label} #{track_id} 的框。"
        rest = (f"範圍外還有 {plan.outside} 個框會保留。" if plan.outside
                else f"⚠ 範圍已涵蓋整條軌跡,{label} #{track_id} 會從整份資料集消失。")
        return plan.box_count, (
            f"命中 {len(plan.frame_indexes)} 幀 / {plan.box_count} 個框。\n{rest}")

    def _delete_track_range(self) -> None:
        """跳出對話框,把指定的那條軌跡在選取的那幾幀裡整段刪掉。

        用途是清幽靈框——目標離場後 tracker 還吐一串,或某一段被誤配到別的目標。
        這種錯誤總是「一整段」,逐幀點框刪在幾百幀的序列上慢到沒人會做。

        目標與影響範圍都在對話框裡即時連動,不靠背後的隱性記憶:最後點過的框只
        當**預填值**,人送出去的永遠是畫面上那一組。

        前置不足時一律說明原因(同 ``_interpolate`` / ``_remap_track``):靜默失敗
        會讓人以為程式壞了。
        """
        if not self.frames:
            return
        rows = self.list_panel.selected_rows()
        if not rows:
            self.statusBar().showMessage(
                "先在左邊幀清單選要清掉的那幾幀(Shift 拉連續 / Ctrl 加點零散的)", 6000)
            return

        seqs = [self.frames[i].seq for i in rows]
        dialog = DeleteTrackDialog(
            f"選取 {len(rows)} 幀(seq {min(seqs)}..{max(seqs)})",
            lambda label, track_id: self._delete_preview(rows, label, track_id),
            self._last_track,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.target()
        if target is None:
            return
        label, track_id = target
        plan = plan_track_delete(self.frames, label, track_id, rows)
        if plan.targets:
            self.apply_track_delete(plan, label, track_id)

    def apply_track_delete(self, plan: TrackDelete, label: str, track_id: int) -> None:
        """套用一份刪除計畫。與 ``_delete_track_range`` 的對話框分開,方便直接驗收。"""
        touched = plan.frame_indexes
        if not touched:
            return
        self._snapshot_bulk(touched, f"刪除 {label} #{track_id}")
        # 用物件 identity 過濾整份 dets,而不是逐個 index 刪:index 會隨著刪除位移,
        # 而值比對分不出同一幀裡兩個欄位完全相同的重疊框(清單上的 DUP)——那正是
        # 這功能要清掉的狀況之一,漏掉一個等於白刪。
        doomed = {id(det) for _, det in plan.targets}
        with self._frozen_track_memory():
            for index in touched:
                dets = self.frames[index].dets
                dets[:] = [d for d in dets if id(d) not in doomed]
                self.undo_stacks[index].commit(self.frames[index].snapshot())

            self._after_bulk_change(touched)
        # 焦點交還畫布:剛刪掉一整段,下一步一定是看畫面確認刪對了沒。順帶把
        # A / D 翻頁還原回來——人在幀清單裡操作時 _goto 刻意不搶焦點,而清單只吃
        # ↑↓,不放回去的話得先點一下畫布才能繼續用習慣的鍵翻幀。
        self.canvas.setFocus()
        # 預填值顯式設回剛刪的那條:遞補選取已被上面的凍結擋住,這裡處理的是另一
        # 半——對話框裡可以當場改掉 label / id,送出的未必是剛剛點的那條。幽靈框常
        # 分成好幾段,「刪完一段接著刪下一段」是主要用法,不設回去就得重打號碼。
        # 「沿用 #id」刻意不跟著設:它答的是「接著畫的新框掛誰」,而剛刪掉的軌跡
        # 不會馬上補回來,停在使用者最後主動點的那條才對。
        self._last_track = (label, track_id)

        seqs = [self.frames[i].seq for i in touched]
        self.statusBar().showMessage(
            f"已刪掉 {label} #{track_id} 的 {plan.box_count} 個框"
            f"({len(touched)} 幀,seq {min(seqs)}..{max(seqs)}),Ctrl+Shift+I 可整組復原",
            6000,
        )

    # ------------------------------------------------------- 跨幀批次改動的復原

    def _snapshot_bulk(self, touched: list[int], what: str) -> None:
        self._last_bulk = {i: self.frames[i].snapshot() for i in touched}
        self._last_bulk_what = what

    def _undo_last_bulk(self) -> None:
        if not self._last_bulk:
            self.statusBar().showMessage("沒有可復原的批次改動", 3000)
            return
        touched = sorted(self._last_bulk)
        what = self._last_bulk_what
        for index, snapshot in self._last_bulk.items():
            self.frames[index].restore(snapshot)
            self.undo_stacks[index].commit(self.frames[index].snapshot())
        self._last_bulk = {}
        self._last_bulk_what = ""
        self._after_bulk_change(touched)
        self.statusBar().showMessage(f"已復原「{what}」({len(touched)} 幀)", 4000)

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

    def _delete_selected(self) -> None:
        """刪掉選取的框。畫布的 Delete 鍵與選單都走這裡,不各自呼叫 canvas。

        單一入口才擋得住黏著記憶被污染:``canvas.delete_selected`` 刪完會把選取
        遞補到隔壁的框,而「沿用 #id」記的是使用者主動點過誰,不該被這個位移改掉。
        """
        with self._frozen_track_memory():
            self.canvas.delete_selected()

    def _on_selection_changed(self, index: int) -> None:
        frame = self.frames[self.index] if 0 <= self.index < len(self.frames) else None
        selected = self.canvas.selected_det
        self.det_panel.set_det(selected, frame.size if frame is not None else None)
        self._remember_track()
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
        # 在右側面板改掉選取框的 label / track_id 不會發 selectionChanged,兩個
        # 黏著記憶都得在這裡跟上:否則「刪這條」刪的還是改號前的舊軌跡,「沿用
        # #id」也還掛著舊號。
        self._remember_track()
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
        self.act_first.setEnabled(has_data and self.index > 0)
        self.act_last.setEnabled(has_data and self.index < len(self.frames) - 1)
        self.act_save.setEnabled(has_data)
        self.act_fit.setEnabled(has_data)
        self.act_undo.setEnabled(bool(stack and stack.can_undo))
        self.act_redo.setEnabled(bool(stack and stack.can_redo))
        self.act_delete.setEnabled(self.canvas.selected_det is not None)
        # 刻意不依「已勾選 / 已選框」停用:停用的動作按下去毫無反應,
        # 使用者無從得知少了哪一步。改由 _interpolate / _remap_track 逐項說明原因。
        self.act_interp.setEnabled(has_data)
        self.act_remap.setEnabled(has_data)
        self.act_delete_track.setEnabled(has_data)
        self.act_remap_range.setEnabled(has_data)
        self.act_bulk_undo.setEnabled(bool(self._last_bulk))
        for action in (self.act_find, self.act_find_next, self.act_find_prev):
            action.setEnabled(has_data)

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
