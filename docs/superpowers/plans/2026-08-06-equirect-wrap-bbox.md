# equirect 跨接縫 bbox 標註 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓標註工具能表達、繪製與編輯跨越 equirect 左右接縫的 bbox,並停止把上游送來的跨縫框 clamp 成半個框。

**Architecture:** 把「連續表示」(記憶體中的 `det.bbox`,`x1 < x2`、x 可越界)與「落檔表示」(JSON,`x1 ∈ [0,1)`、`x2 = x1 + w` 可 > 1)分開,轉換只發生在 `canonical_bbox`。載入時不需轉換 —— 落檔表示本身就是合法的連續表示。因此拖曳、縮放、hit-test、內插全是普通矩形運算,環繞只出現在繪製、hit-test 的整數圈平移與存檔正規化三處。

**Tech Stack:** Python 3.11+、PyQt6 ≥6.7、ruff(line-length 100、target py311)。無新增依賴。

## Global Constraints

- **JSON schema 一個字都不改**:仍是四個 float、欄位與 key 順序原樣、存檔契約前三條(只替換 `raw["dets"]`、det 只覆寫四欄、行尾/BOM/結尾換行原樣)不變。
- **不改 `D:\ws\detect_stream` 的任何檔案**。只讀取它的資料與 `wrap_iou` 做驗證。
- **y 軸行為完全不變**:equirect 上下是極點、不是環狀鄰接。所有 clamp `[0,1]`、`_ensure_span` 的 y 分支原樣保留。
- **`wrap=False` 時所有行為與改動前逐位元相同**。非 2:1 的資料不得受任何影響。
- **驗收不得寫死資料量**:標註正在進行中,幀數/框數/跨縫框個數隨時在變。只斷言與資料量無關的性質。
- **ruff line-length 100**,提交前必須 `uv run ruff check .` 通過。
- 每個 task 結束前跑 `uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py`,必須全部通過才 commit。
- commit 訊息格式:`<type>(labeling): <繁中 subject,說為什麼>`,結尾加 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。

---

## File Structure

| 檔案                        | 這次負責什麼                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| `gt_labeling/model.py`      | `canonical_bbox` 的 wrap 契約、`FrameLabel.wrap_x` / `is_equirect` / `has_edge_box`、內插最短弧 |
| `gt_labeling/transform.py`  | `ViewTransform.wrap_x`、`visible_shifts`、`clamp_offset` 的 x 取模                              |
| `gt_labeling/canvas.py`     | 影像鋪排繪製、框與 hit-test 的整數圈平移、編輯時 x 不夾                                         |
| `gt_labeling/window.py`     | 2:1 自動判定、檢視選單開關、狀態列、切換時同步 model 與 canvas                                  |
| `gt_labeling/panel.py`      | 幀清單 `CUT` 旗標與摘要計數                                                                     |
| `tests/verify_roundtrip.py` | 座標契約、存檔往返、內插、`ViewTransform` 的驗收                                                |
| `tests/verify_gui.py`       | 環景畫布的離屏互動驗收                                                                          |
| `README.md`                 | 存檔契約第 4 條、操作說明                                                                       |

---

### Task 1: `canonical_bbox` 的 wrap 契約

**Files:**

- Modify: `gt_labeling/model.py:33-52`
- Test: `tests/verify_roundtrip.py`(`test_canonical_bbox`)

**Interfaces:**

- Consumes: 無(這是最底層)
- Produces: `canonical_bbox(bbox, wrap: bool = False) -> list[float]`。所有後續 task 呼叫它時都要決定 `wrap` 的值。

- [ ] **Step 1: 寫失敗測試**

在 `tests/verify_roundtrip.py` 的 `test_canonical_bbox()` 尾端(第 272 行 `"canonical 是冪等的"` 那個 `check` 之後)追加:

```python
    section("canonical_bbox 環景模式(wrap=True)")
    # 跨縫框:上游 gt_densify 送來的形式,不得被 clamp 成半個框
    check(canonical_bbox([0.94063, 0.5, 1.00348, 0.7], wrap=True)
          == [0.94063, 0.5, 1.00348, 0.7], "跨縫框原樣保留,x2 不被夾到 1.0")
    check(canonical_bbox(canonical_bbox([0.94063, 0.5, 1.00348, 0.7], wrap=True), wrap=True)
          == canonical_bbox([0.94063, 0.5, 1.00348, 0.7], wrap=True),
          "跨縫框的 canonical 是冪等的")
    # 一般框在 wrap 下輸出必須與 wrap=False 完全相同
    for bbox in ([0.3, 0.4, 0.5, 0.6], [0.12346, 0.5, 0.98765, 0.6]):
        check(canonical_bbox(bbox, wrap=True) == canonical_bbox(bbox, wrap=False),
              f"非跨縫框 {bbox} 在 wrap 下輸出不變")
    # 往左出界:x1 取模後自然落回右側
    check(canonical_bbox([-0.03, 0.5, 0.02, 0.7], wrap=True) == [0.97, 0.5, 1.02, 0.7],
          "往左出界的框 x1 取模成 0.97,寬度保持 0.05")
    # 越過一整圈以上:x1 仍落回 [0,1)
    left = canonical_bbox([1.94063, 0.5, 1.96, 0.7], wrap=True)
    check(0.0 <= left[0] < 1.0 and round(left[2] - left[0], 5) == 0.01937,
          f"繞超過一圈的 x1 落回 [0,1) 且寬度不變 {left}")
    # 反向拖曳仍被排序(跨縫由 x2 越界表達,不由 x1>x2 表達)
    check(canonical_bbox([0.6, 0.7, 0.2, 0.3], wrap=True) == [0.2, 0.3, 0.6, 0.7],
          "wrap 下反向座標一樣會被排序")
    # 寬度上限:超過一整圈夾成一圈
    full = canonical_bbox([0.1, 0.2, 2.5, 0.4], wrap=True)
    check(round(full[2] - full[0], 5) == 1.0, f"寬度超過一整圈被夾成 1.0 {full}")
    # 退化框
    tiny = canonical_bbox([0.5, 0.5, 0.5, 0.5], wrap=True)
    check(tiny[0] < tiny[2] and tiny[1] < tiny[3], f"wrap 下退化框仍保證非零寬高 {tiny}")
    # 貼右緣的退化框:wrap 下往右長,不需要往左長的 fallback
    edge = canonical_bbox([1.0, 0.5, 1.0, 0.5], wrap=True)
    check(edge[0] == 0.0 and edge[2] > 0.0, f"x1=1.0 取模成 0.0 並往右長 {edge}")
    # y 軸行為在 wrap 下完全不變
    check(canonical_bbox([0.3, -0.5, 0.4, 1.8], wrap=True)[1::2] == [0.0, 1.0],
          "wrap 下 y 仍 clamp 到 [0,1]")
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: `TypeError: canonical_bbox() got an unexpected keyword argument 'wrap'`(腳本直接拋例外中止)。

- [ ] **Step 3: 實作**

把 `gt_labeling/model.py:33-52` 整段替換成:

```python
def canonical_bbox(bbox, wrap: bool = False) -> list[float]:
    """排序 -> clamp -> round 5 位 -> 保證有非零寬高。

    ``wrap=True`` 是 equirect 環景模式:x 不 clamp,改成把左界取模回 ``[0,1)``、
    右界寫成 ``x1 + 寬度``(可越過 1.0)。這個「延伸表示」不是本工具發明的 —— 上游
    ``gt_densify.py`` 就這樣落檔(實測 x2 到 1.11),下游 ``eval_gt.py`` /
    ``evaluate.py`` 的 ``wrap_iou`` 靠 x 平移 ±1 來配對它。夾回 [0,1] 會把跨縫的人
    切成半個框,而且沒有任何流程會報錯。

    跨縫由「x2 越界」表達,不由「x1 > x2」表達,所以這裡照樣排序 —— 反向拖曳
    (往起點左邊拖)的既有行為因此完全不受影響。``wrap_iou`` 也吃不下 x2<x1:
    那會算成負寬、面積 0,該框在評估裡永遠是 FN。

    y 兩種模式都一樣 clamp 到 [0,1]:equirect 的上下是極點,不是環狀鄰接。
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    y1, y2 = (min(max(v, 0.0), 1.0) for v in (y1, y2))
    y1, y2 = (round(v, BBOX_DP) for v in (y1, y2))
    y1, y2 = _ensure_span(y1, y2)

    if wrap:
        # 先 round 再取模:round(0.999996, 5) 會變成 1.0,取模後才落回 0.0;
        # 反過來先取模的話,0.999996 會原樣留下,再 round 就跑出 [0,1) 了。
        width = min(max(x2 - x1, MIN_SPAN), 1.0)
        x1 = round(round(x1, BBOX_DP) % 1.0, BBOX_DP)
        x2 = round(x1 + width, BBOX_DP)
    else:
        x1, x2 = (min(max(v, 0.0), 1.0) for v in (x1, x2))
        x1, x2 = (round(v, BBOX_DP) for v in (x1, x2))
        x1, x2 = _ensure_span(x1, x2)
    return [x1, y1, x2, y2]
```

`_ensure_span` 保持原樣不動 —— wrap 分支不呼叫它(x 沒有 1.0 上限,直接 `max(width, MIN_SPAN)` 就夠)。

- [ ] **Step 4: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 「全部通過」,ruff 無錯。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/model.py tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
feat(labeling): canonical_bbox 支援環景延伸表示,跨縫框不再被夾成半個

equirect 的 x=0 與 x=1 是同一條經線。wrap=True 時 x 改成「左界取模回
[0,1)、右界寫成 x1+寬度」,與上游 gt_densify 落檔、下游 wrap_iou 消費的
表示一致。wrap=False 逐位元維持原行為。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `FrameLabel` 的環景狀態與存檔傳遞

**Files:**

- Modify: `gt_labeling/model.py:55-92`(`Det.to_json`)、`111-135`(`FrameLabel` 欄位與 property)、`171-172`(`dets_json`)、`202-225`(`load_frame`)
- Test: `tests/verify_roundtrip.py`

**Interfaces:**

- Consumes: `canonical_bbox(bbox, wrap)`(Task 1)
- Produces:
  - `Det.to_json(wrap: bool = False) -> dict`
  - `FrameLabel.wrap_x: bool`(dataclass 欄位,**不是** JSON 欄位)
  - `FrameLabel.is_equirect -> bool`
  - `FrameLabel.dets_json() -> list[dict]`(內部改用 `self.wrap_x`,簽名不變)
  - `load_frame(path)` 回傳的 frame 其 `wrap_x` 已依 size 自動設好

- [ ] **Step 1: 寫失敗測試**

在 `tests/verify_roundtrip.py` 的 `test_canonical_bbox()` **之後**新增一個函式:

```python
def test_frame_wrap_state() -> None:
    section("FrameLabel 的環景狀態")

    def frame_with(size: list[int], bboxes: list[list[float]]) -> FrameLabel:
        raw = {"type": "gt", "version": 1, "seq": 1, "size": size, "dets": []}
        dets = [Det(label="person", track_id=1, ppe="ng", bbox=list(b)) for b in bboxes]
        frame = FrameLabel(path=Path("000001.json"), raw=raw, dets=dets, style=TextStyle())
        frame.wrap_x = frame.is_equirect
        frame._clean = frame.dets_json()
        return frame

    erp = frame_with([3840, 1920], [[0.94063, 0.5, 1.00348, 0.7]])
    check(erp.is_equirect, "3840x1920 判定為 equirect")
    check(erp.wrap_x, "equirect 的 frame 預設進環景模式")
    check(erp.dets_json()[0]["bbox"] == [0.94063, 0.5, 1.00348, 0.7],
          f"環景下跨縫框存出去原樣保留 {erp.dets_json()[0]['bbox']}")
    check(not erp.dirty, "跨縫框讀進來不會被誤判為已改")

    flat = frame_with([1920, 1080], [[0.94063, 0.5, 1.00348, 0.7]])
    check(not flat.is_equirect, "1920x1080 不是 equirect")
    check(not flat.wrap_x, "非 equirect 的 frame 不進環景模式")
    check(flat.dets_json()[0]["bbox"] == [0.94063, 0.5, 1.0, 0.7],
          f"非環景下 x2 仍被 clamp 到 1.0(既有行為){flat.dets_json()[0]['bbox']}")

    erp.wrap_x = False
    check(erp.dirty, "手動關掉環景模式後,含跨縫框的 frame 被標成已改(存出去會不同)")

    plain = frame_with([3840, 1920], [[0.3, 0.4, 0.5, 0.6]])
    plain_json = plain.dets_json()
    plain.wrap_x = False
    check(plain.dets_json() == plain_json, "沒有跨縫框的 frame,切換模式輸出完全相同")
```

並在 `main()` 的 `test_canonical_bbox()` 呼叫之後(第 340 行)加一行:

```python
    test_frame_wrap_state()
```

同時修正 `test_edit_roundtrip` 第 159-162 行那個會與新契約衝突的斷言,把:

```python
    all_normalized = all(
        0.0 <= v <= 1.0 for d in saved_raw["dets"] for v in d["bbox"]
    )
    check(all_normalized, "存出的 bbox 全在 [0,1](仍是歸一化,不是像素)")
```

換成:

```python
    # 環景模式下 x2 可以越過 1.0(跨縫框的延伸表示),但 x1 必須落在 [0,1)、
    # y 必須落在 [0,1]。寫成「x2 一律 <= 1」會在跨縫框上假 FAIL。
    wrap = load_frame(target.label_path).wrap_x
    x_ok = all(
        0.0 <= d["bbox"][0] < 1.0 and (d["bbox"][2] <= 1.0 if not wrap else True)
        for d in saved_raw["dets"]
    )
    y_ok = all(
        0.0 <= d["bbox"][1] <= 1.0 and 0.0 <= d["bbox"][3] <= 1.0
        for d in saved_raw["dets"]
    )
    check(x_ok, f"存出的 x1 都在 [0,1)(環景={wrap})")
    check(y_ok, "存出的 y 都在 [0,1]")
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: `AttributeError: 'FrameLabel' object has no attribute 'is_equirect'`。

- [ ] **Step 3: 實作**

3a. `gt_labeling/model.py`,`Det.to_json`(第 78-86 行)改成:

```python
    def to_json(self, wrap: bool = False) -> dict:
        # 以原始 det 當底再覆寫:更新既有 key 不會改變 dict 的順序,所以連
        # key 排列都跟讀進來時一樣;上游多寫的欄位(src 等)也原封不動帶回去。
        out = dict(self.raw)
        out["label"] = self.label
        out["track_id"] = self.track_id
        out["ppe"] = self.ppe if self.is_person else None
        out["bbox"] = canonical_bbox(self.bbox, wrap)
        return out
```

3b. `FrameLabel` 的 dataclass 欄位(第 111-117 行)加一個 `wrap_x`:

```python
@dataclass
class FrameLabel:
    path: Path
    raw: dict
    dets: list[Det]
    style: TextStyle
    # equirect 環景模式:x 以延伸表示落檔(見 canonical_bbox)。**不是** JSON 欄位,
    # 是執行期狀態,由 load_frame 依 size 自動判定、由視窗的檢視選單覆寫。
    wrap_x: bool = False
    _clean: list[dict] = field(default_factory=list, repr=False)
```

3c. 在 `size` property(第 125-130 行)**之後**加兩個 property:

```python
    @property
    def is_equirect(self) -> bool:
        """寬高比 2:1 視為 equirectangular。

        JSON 沒有、也不會加「這是 equirect」的欄位,size 是唯一線索。誤判成環景的
        代價是讓人在 perspective 影像上畫出 x2>1 的框,而下游 wrap_iou 對非環狀來源
        的 ±1 平移不會命中 —— 那種框會靜默算成 FN,所以判定寧可保守。
        """
        try:
            width, height = self.size
        except ValueError:
            return False
        return width == height * 2
```

在 `has_duplicate_track`(第 149-167 行)**之後**加:

```python
    @property
    def has_edge_box(self) -> bool:
        """有框恰好貼在 x=0 或 x=1 —— 環景下這幾乎一定是被 clamp 削過的痕跡。

        用精確相等而非「接近」:clamp 產生的正是這兩個值,而 canonical 後所有座標
        都是 round 到 5 位的十進位值,不存在差一點點的情況。``x2 > 1`` 是健康的跨縫
        框,不算在內。

        模式判斷寫在這裡而不是交給呼叫端:非 equirect 資料貼邊是正常的,恆回 False。
        """
        if not self.wrap_x:
            return False
        return any(d.bbox[0] == 0.0 or d.bbox[2] == 1.0 for d in self.dets)
```

3d. `dets_json`(第 171-172 行)改成:

```python
    def dets_json(self) -> list[dict]:
        return [d.to_json(self.wrap_x) for d in self.dets]
```

3e. `load_frame`(第 222-225 行結尾)改成:

```python
    frame = FrameLabel(path=path, raw=raw, dets=dets, style=style)
    # 先定模式再算乾淨基準:_clean 是用 dets_json() 算的,而 dets_json() 的結果
    # 取決於 wrap_x。順序反了的話,equirect 檔一開啟就會被標成已改。
    frame.wrap_x = frame.is_equirect
    frame._clean = frame.dets_json()
    return frame
```

- [ ] **Step 4: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 「全部通過」。特別確認 `未編輯存檔:byte 完全相同` 與 `det 的未知欄位` 兩節仍全綠 —— 那份樣本(`0625_145125/000-020s`)是 3840x1920 但沒有跨縫框,所以自動進環景模式後輸出必須一模一樣。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/model.py tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
feat(labeling): 2:1 影像自動進環景模式,存檔不再削掉跨縫框

FrameLabel 加執行期欄位 wrap_x(非 JSON 欄位),load_frame 依 size 寬高比
2:1 自動判定,dets_json 據此決定 canonical 的模式。沒有跨縫框的檔案輸出
逐位元不變。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 內插走 ERP 最短弧

**Files:**

- Modify: `gt_labeling/model.py:240-282`(`interpolate_missing`)
- Test: `tests/verify_roundtrip.py`

**Interfaces:**

- Consumes: `canonical_bbox(bbox, wrap)`(Task 1)、`FrameLabel.wrap_x`(Task 2)
- Produces: `interpolate_missing(frames, label, track_id, max_gap) -> Interpolation`(簽名不變,行為依 `frames[0].wrap_x` 改變)、模組級私有函式 `_wrap_align(target, reference) -> list[float]`

- [ ] **Step 1: 寫失敗測試**

在 `tests/verify_roundtrip.py` 的 `test_interpolate_is_per_label()` **之後**新增:

```python
def test_interpolate_wraps_shortest_arc() -> None:
    """跨接縫的兩個錨點之間,補出來的框必須走最短弧,不能橫掃整張圖。

    錨點 A 在 x≈0.97、錨點 B 在 x≈0.02,兩者實際只差 0.05 個畫面寬(人走過接縫)。
    直接對座標線性內插會讓中間的框從右邊一路掃到左邊,產生一串完全錯誤的 GT。
    """
    section("補框:跨接縫走最短弧")

    def frame_of(seq: int, dets: list[Det], wrap: bool) -> FrameLabel:
        raw = {"type": "gt", "version": 1, "seq": seq, "size": [3840, 1920], "dets": []}
        frame = FrameLabel(path=Path(f"{seq:06d}.json"), raw=raw, dets=dets,
                           style=TextStyle())
        frame.wrap_x = wrap
        return frame

    def person(bbox: list[float]) -> Det:
        return Det(label="person", track_id=1, ppe="ng", bbox=bbox)

    # A 在 0.96~0.99,B 在 0.01~0.04(= A 往右走 0.05 圈後的位置)
    frames = [
        frame_of(1, [person([0.96, 0.5, 0.99, 0.7])], wrap=True),
        frame_of(2, [], wrap=True),
        frame_of(3, [person([0.01, 0.5, 0.04, 0.7])], wrap=True),
    ]
    plan = interpolate_missing(frames, "person", 1, max_gap=10)
    check(len(plan.additions) == 1, f"補出 1 個框(實際 {len(plan.additions)})")
    if plan.additions:
        bbox = plan.additions[0][1].bbox
        width = round((bbox[2] - bbox[0]) % 1.0, 5)
        check(width == 0.03, f"補出的框寬度仍是 0.03,沒有被拉長 {bbox}")
        # 中點應落在 0.985 與 1.025 之間,取模後 = 0.985 或 0.005 附近
        centre = ((bbox[0] + bbox[2]) * 0.5) % 1.0
        check(centre > 0.98 or centre < 0.02,
              f"中點落在接縫附近而不是畫面中央 {centre:.5f}")

    # 非環景模式下維持原本的逐座標內插(不得偷偷改變既有行為)
    flat = [
        frame_of(1, [person([0.96, 0.5, 0.99, 0.7])], wrap=False),
        frame_of(2, [], wrap=False),
        frame_of(3, [person([0.01, 0.5, 0.04, 0.7])], wrap=False),
    ]
    plan_flat = interpolate_missing(flat, "person", 1, max_gap=10)
    check(plan_flat.additions and plan_flat.additions[0][1].bbox == [0.485, 0.5, 0.515, 0.7],
          f"非環景維持原本的直線內插 "
          f"{plan_flat.additions[0][1].bbox if plan_flat.additions else None}")
```

在 `main()` 的 `test_interpolate_is_per_label()` 呼叫之後加一行:

```python
    test_interpolate_wraps_shortest_arc()
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: `FAIL 補出的框寬度仍是 0.03,沒有被拉長 [0.485, 0.5, 0.515, 0.7]` —— 中點跑到畫面中央,寬度看似正常但位置完全錯。

- [ ] **Step 3: 實作**

在 `gt_labeling/model.py` 的 `interpolate_missing` **之前**加一個私有函式:

```python
def _wrap_align(target: list[float], reference: list[float]) -> list[float]:
    """把 ``target`` 的 x 平移整數圈,使其中心離 ``reference`` 中心最近。

    equirect 的 x 是環狀的:0.02 與 0.97 相隔 0.05,不是 0.95。兩個錨點各自被
    正規化到 [0,1) 之後,這個資訊就只能靠「差幾圈」還原。
    """
    centre_t = (target[0] + target[2]) * 0.5
    centre_r = (reference[0] + reference[2]) * 0.5
    shift = round(centre_r - centre_t)
    if shift == 0:
        return list(target)
    return [target[0] + shift, target[1], target[2] + shift, target[3]]
```

然後把 `interpolate_missing` 的內插迴圈(原第 267-282 行)改成:

```python
    wrap = bool(frames) and frames[0].wrap_x
    result = Interpolation()
    for (i0, s0, d0), (i1, s1, d1) in zip(anchors, anchors[1:], strict=False):
        if i1 - i0 <= 1:
            continue
        span = s1 - s0
        if span > max_gap:
            result.skipped.append((s0, s1, span))
            continue
        # 環景:先把後錨點平移到前錨點的同一圈,線性內插才走最短弧。
        end = _wrap_align(d1.bbox, d0.bbox) if wrap else d1.bbox
        for k in range(i0 + 1, i1):
            t = (frames[k].seq - s0) / max(span, 1)
            bbox = [d0.bbox[m] + (end[m] - d0.bbox[m]) * t for m in range(4)]
            result.additions.append(
                (k, Det(label=d0.label, track_id=track_id, ppe=d0.ppe,
                        bbox=canonical_bbox(bbox, wrap)))
            )
    return result
```

同時在 `interpolate_missing` 的 docstring 末尾補一段:

```python
    equirect 資料(``frames[0].wrap_x``)的 x 是環狀的:後錨點先被平移到前錨點的同一圈
    再內插,否則一個走過接縫的人會補出一串從畫面右邊橫掃到左邊的假框。
```

- [ ] **Step 4: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 「全部通過」,含既有的 `補框:person#1 與 drone#1 不得互相干擾` 全綠。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/model.py tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
fix(labeling): 補框在環景下走最短弧,跨接縫不再補出橫掃整張圖的假框

錨點 A 在 x≈0.97、B 在 x≈0.02 實際只差 0.05 圈,逐座標線性內插會讓中間
的框從右邊掃到左邊。內插前先把後錨點平移到前錨點同一圈,作法與上游
gt_densify.interp_box 一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `ViewTransform` 的環繞座標

**Files:**

- Modify: `gt_labeling/transform.py:13-30`(imports 與欄位)、`114-126`(`clamp_offset`)
- Test: `tests/verify_roundtrip.py`

**Interfaces:**

- Consumes: 無
- Produces:
  - `ViewTransform.wrap_x: bool`(dataclass 欄位,預設 `False`)
  - `ViewTransform.visible_shifts(view_w: float) -> range` —— 與視窗相交的整數圈編號。非環繞恆回 `range(0, 1)`。
  - `clamp_offset(view)` 在 `wrap_x` 時對 x 取模而非夾住

- [ ] **Step 1: 寫失敗測試**

在 `tests/verify_roundtrip.py` 的 `test_transform_roundtrip()` **之後**新增:

```python
def test_transform_wrap() -> None:
    section("ViewTransform 環繞:整數圈與 offset 取模")
    tf = ViewTransform()
    tf.set_image_size(3840, 1920)
    tf.wrap_x = True
    tf.zoom = 1000.0 / 3840.0          # span_x = 1000
    check(round(tf.span_x, 6) == 1000.0, f"span_x = {tf.span_x}")

    tf.off_x = 0.0
    check(list(tf.visible_shifts(1600.0)) == [0, 1],
          f"影像左緣貼齊視窗左邊,需要 2 份 {list(tf.visible_shifts(1600.0))}")
    tf.off_x = -500.0
    check(list(tf.visible_shifts(1600.0)) == [0, 1, 2],
          f"往左捲半份,需要 3 份 {list(tf.visible_shifts(1600.0))}")
    tf.off_x = 300.0
    check(list(tf.visible_shifts(1600.0)) == [-1, 0, 1],
          f"往右捲時左邊要補一份 {list(tf.visible_shifts(1600.0))}")

    # 放大到單份就蓋滿視窗時只需要一份
    tf.zoom = 10000.0 / 3840.0
    tf.off_x = -3000.0
    check(list(tf.visible_shifts(1600.0)) == [0],
          f"單份蓋滿視窗只畫一份 {list(tf.visible_shifts(1600.0))}")

    # 非環繞恆為一份,且 clamp_offset 維持原本的夾住行為
    flat = ViewTransform()
    flat.set_image_size(3840, 1920)
    flat.zoom = 1000.0 / 3840.0
    flat.off_x = 999.0
    check(list(flat.visible_shifts(1600.0)) == [0], "非環繞恆回 1 份")
    flat.clamp_offset(QSize(1600, 900))
    check(flat.off_x == 300.0,
          f"非環繞、影像比視窗小 → 置中到 300(既有行為,實際 {flat.off_x})")

    # 環繞下 pan 不被夾住,只被取模;取模後畫面內容不變(靠補圈)
    tf2 = ViewTransform()
    tf2.set_image_size(3840, 1920)
    tf2.wrap_x = True
    tf2.zoom = 1000.0 / 3840.0
    tf2.off_x = 4321.0
    tf2.clamp_offset(QSize(1600, 900))
    check(0.0 <= tf2.off_x < tf2.span_x, f"off_x 被取模回 [0, span_x) {tf2.off_x}")
    check(round(tf2.off_x, 6) == 321.0, f"4321 取模 1000 = 321(實際 {tf2.off_x})")

    # y 方向在環繞下維持原本的夾住行為
    tf2.zoom = 2.0
    tf2.off_y = 5000.0
    tf2.clamp_offset(QSize(1600, 900))
    check(tf2.off_y == 0.0, f"y 仍被夾住不許拖出視窗(實際 {tf2.off_y})")
```

在 `main()` 的 `test_transform_roundtrip()` 呼叫之後加一行:

```python
    test_transform_wrap()
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: `AttributeError: 'ViewTransform' object has no attribute 'wrap_x'`。

- [ ] **Step 3: 實作**

3a. `gt_labeling/transform.py` 第 13-17 行的 import 區加入 `math`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, QSize, QSizeF
```

3b. dataclass 欄位(第 23-29 行)加 `wrap_x`:

```python
@dataclass
class ViewTransform:
    img_w: int = 1
    img_h: int = 1
    zoom: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0
    # equirect 環景:x 方向無限環繞(影像左右重複鋪排、pan 不夾)。y 不環繞。
    wrap_x: bool = False
```

3c. 在 `image_rect`(第 78-79 行)**之後**加:

```python
    def visible_shifts(self, view_w: float) -> range:
        """x 環繞時,與視窗相交的整數圈編號;非環繞恆為 ``range(0, 1)``。

        第 k 圈的影像佔 widget 的 ``[off_x + k*span_x, off_x + (k+1)*span_x]``。
        繪製與 hit-test 都經由這裡取圈數,環繞語意因此只有一個出處 —— 與「座標
        換算只在 ViewTransform」的既有不變量同一個理由。
        """
        if not self.wrap_x or self.span_x <= 0.0:
            return range(0, 1)
        k_min = math.floor(-self.off_x / self.span_x - 1.0) + 1
        k_max = math.ceil((view_w - self.off_x) / self.span_x) - 1
        return range(k_min, k_max + 1)
```

3d. `clamp_offset`(第 114-126 行)的 x 分支改成:

```python
    def clamp_offset(self, view: QSize) -> None:
        """影像比視窗小則置中,比視窗大則不許拖出視窗外。

        環景模式的 x 是例外:不夾、只取模。夾住的話接縫永遠停在視窗邊緣,人就沒辦法
        把它拉到中間畫框;取模則順便擋掉長時間平移累積的浮點漂移,而畫面內容不受影響
        —— 少掉的那一圈由 :meth:`visible_shifts` 補回來。
        """
        if view.width() > 0:
            if self.wrap_x:
                self.off_x %= self.span_x
            elif self.span_x <= view.width():
                self.off_x = (view.width() - self.span_x) / 2.0
            else:
                self.off_x = min(max(self.off_x, view.width() - self.span_x), 0.0)
        if view.height() > 0:
            if self.span_y <= view.height():
                self.off_y = (view.height() - self.span_y) / 2.0
            else:
                self.off_y = min(max(self.off_y, view.height() - self.span_y), 0.0)
```

- [ ] **Step 4: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 「全部通過」,含既有的 `ViewTransform 可逆性(座標漂移防線)` 全綠。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/transform.py tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
feat(labeling): ViewTransform 支援 x 環繞,接縫可以拉到畫面中間

新增 wrap_x 與 visible_shifts:環景時 pan 不夾只取模,少掉的那一圈由
visible_shifts 算出來補畫。環繞語意集中在 transform 一處,維持「座標換算
只有一個出處」的既有不變量。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 環景畫布 —— 繪製、hit-test、編輯夾限

**Files:**

- Modify: `gt_labeling/canvas.py:46-54`(顏色常數)、`86-87`(`_clamp01`)、`236-306`(hit-test 與編輯)、`373-374` 與 `396-399`(新框)、`500-533`(繪製影像)、`557-600`(繪製框)
- Test: `tests/verify_roundtrip.py`(幾何前提)、`tests/verify_gui.py`(hit-test 紅燈)

**Interfaces:**

- Consumes: `ViewTransform.wrap_x` / `visible_shifts`(Task 4)、`FrameLabel.wrap_x`(Task 2)、`canonical_bbox(bbox, wrap)`(Task 1)
- Produces: `ImageCanvas.set_wrap_x(enabled: bool) -> None`。canvas 內部一律以 `self._wrap()`(讀 `self._frame.wrap_x`)決定存檔語意,`self.tf.wrap_x` 只管檢視。

- [ ] **Step 1a: 寫幾何前提測試**

在 `tests/verify_roundtrip.py` 的 `test_transform_wrap()` **之後**新增(純幾何,不需要 Qt 視窗):

```python
def test_canvas_wrap_geometry() -> None:
    """畫布環繞的兩個關鍵幾何:框在每一圈的位置、hit-test 命中對側那一份。"""
    section("環景畫布幾何")
    tf = ViewTransform()
    tf.set_image_size(3840, 1920)
    tf.wrap_x = True
    tf.zoom = 1000.0 / 3840.0          # span_x = 1000
    tf.off_x = 0.0
    tf.off_y = 0.0

    # 跨縫框 x=0.94~1.01:主圈畫在 940~1010,前一圈畫在 -60~10
    bbox = [0.94, 0.5, 1.01, 0.7]
    rect = tf.n2v_rect(bbox)
    check(round(rect.left(), 6) == 940.0 and round(rect.right(), 6) == 1010.0,
          f"主圈落在 940~1010 {rect.left()}~{rect.right()}")
    shifted = rect.translated(-1 * tf.span_x, 0.0)
    check(round(shifted.left(), 6) == -60.0 and round(shifted.right(), 6) == 10.0,
          f"前一圈落在 -60~10 {shifted.left()}~{shifted.right()}")

    # 視窗寬 1600:x=5 這個點應該落在「前一圈」的框裡,主圈測不到
    point_x = 5.0
    hit_main = rect.left() <= point_x <= rect.right()
    hit_prev = shifted.left() <= point_x <= shifted.right()
    check(not hit_main and hit_prev,
          "畫面最左邊的點命中的是前一圈的框,所以 hit-test 必須逐圈測")
    check(-1 in tf.visible_shifts(1600.0) or 0 in tf.visible_shifts(1600.0),
          f"visible_shifts 有涵蓋這些圈 {list(tf.visible_shifts(1600.0))}")
```

在 `main()` 的 `test_transform_wrap()` 呼叫之後加一行:

```python
    test_canvas_wrap_geometry()
```

這一段只用 Task 4 已完成的 API,寫完就會通過。它的作用是把「hit-test 必須逐圈測」這個要求鎖成可執行的斷言 —— 真正的紅燈在 Step 1b。

- [ ] **Step 1b: 寫會失敗的 hit-test 測試**

在 `tests/verify_gui.py` 的 `main()` 裡,`section("開資料夾")` 那一段最後一個 `check`(`"首幀尺寸 3840x1920"`)**之後**插入:

```python
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
        canvas.tf.wrap_x = False
        app.processEvents()
```

`verify_gui.py` 需要 `Det`、`QTest`、`QPoint`、`Qt` —— 前三者確認 import 區已有(`drag()` 與 `find_empty_spot()` 都用到 `QTest` / `QPoint` / `Qt`);`Det` 若尚未匯入,在 `from gt_labeling.model import ...` 那行補上。

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling python tests/verify_gui.py
```

Expected:

- `verify_roundtrip.py` 全部通過(幾何前提用的是 Task 4 的 API)。
- `verify_gui.py` **FAIL** `從畫面左緣點到繞過來的那一段,選中同一個框(預期 index N,實際 -1)` —— 現行 `_hit_box` 只測主圈,畫面左緣那一段點不到。

- [ ] **Step 3: 實作**

3a. 顏色常數(`canvas.py` 第 54 行之後)加一個接縫線顏色:

```python
COLOR_IMAGE_EDGE = QColor("#3c3c3c")
COLOR_SEAM = QColor(79, 163, 255, 90)
```

3b. `_clamp01`(第 86-87 行)換成兩個函式:

```python
def _clamp01(p: QPointF) -> QPointF:
    return QPointF(min(max(p.x(), 0.0), 1.0), min(max(p.y(), 0.0), 1.0))


def _clamp_y_only(p: QPointF) -> QPointF:
    """環景:x 可以越界(跨縫的表達方式),y 仍夾在畫面內(上下是極點)。"""
    return QPointF(p.x(), min(max(p.y(), 0.0), 1.0))
```

3c. 在 `_set_selection`(第 215 行)**之前**加兩個內部工具:

```python
    def _wrap(self) -> bool:
        """存檔語意以 model 為準:frame.wrap_x 決定 canonical 怎麼寫。"""
        return self._frame is not None and self._frame.wrap_x

    def _shifts(self) -> range:
        return self.tf.visible_shifts(float(self.width()))
```

3d. 在 `set_band`(第 180-183 行)**之後**加對外入口:

```python
    def set_wrap_x(self, enabled: bool) -> None:
        """切換環景檢視。frame.wrap_x 由視窗負責設定,這裡只同步檢視與重繪。"""
        if self.tf.wrap_x == enabled:
            return
        self.tf.wrap_x = enabled
        self.tf.clamp_offset(self.size())
        self.viewChanged.emit()
        self.update()
```

並在 `set_frame`(第 159-167 行)裡,`self.tf.set_image_size(...)` 之後、`if prev_size != ...` 之前插入一行,讓換資料夾時檢視跟著 model 走:

```python
            self.tf.wrap_x = frame.wrap_x
```

3e. hit-test 逐圈測 —— `_hit_handle`(第 236-245 行)與 `_hit_box`(第 253-263 行)改成:

```python
    def _hit_handle(self, pos: QPointF) -> int | None:
        det = self.selected_det
        if det is None:
            return None
        base = self.tf.n2v_rect(det.bbox)
        for k in self._shifts():
            rect = base.translated(k * self.tf.span_x, 0.0)
            for i, (hx, hy) in enumerate(HANDLES):
                c = self._handle_center(rect, hx, hy)
                if abs(pos.x() - c.x()) <= HANDLE_HIT and abs(pos.y() - c.y()) <= HANDLE_HIT:
                    return i
        return None

    def _hit_box(self, pos: QPointF) -> int | None:
        if self._frame is None:
            return None
        dets = self._frame.dets
        # 已選取的優先(拖曳穩定),其餘由上層(繪製順序在後)往下找。
        order = [self._sel] if 0 <= self._sel < len(dets) else []
        order += range(len(dets) - 1, -1, -1)
        shifts = list(self._shifts())
        for i in order:
            base = self._hit_rect(dets[i].bbox)
            if any(base.translated(k * self.tf.span_x, 0.0).contains(pos) for k in shifts):
                return i
        return None
```

3f. 編輯時 x 不夾 —— `_apply_move`(第 273-285 行)與 `_apply_resize`(第 287-306 行)改成:

```python
    def _apply_move(self, pos: QPointF) -> None:
        det = self.selected_det
        if det is None:
            return
        now = self.tf.v2n_point(pos)
        dx = now.x() - self._drag_n0.x()
        dy = now.y() - self._drag_n0.y()
        x1, y1, x2, y2 = self._orig_bbox
        # 整體平移不變形:位移量夾到框仍在 [0,1] 內。環景的 x 例外 —— 框本來就
        # 該能走過接縫,夾住等於禁止跨縫。
        if not self._wrap():
            dx = min(max(dx, -x1), 1.0 - x2)
        dy = min(max(dy, -y1), 1.0 - y2)
        det.bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        self.update()

    def _apply_resize(self, pos: QPointF) -> None:
        det = self.selected_det
        if det is None:
            return
        now = self.tf.v2n_point(pos)
        dx = now.x() - self._drag_n0.x()
        dy = now.y() - self._drag_n0.y()
        hx, hy = HANDLES[self._handle]
        x1, y1, x2, y2 = self._orig_bbox
        wrap = self._wrap()
        # 每次都從「拖曳起點的原始 bbox」重算,不累加,避免捨入誤差堆積。
        if hx < 0:
            x1 = x1 + dx if wrap else min(max(x1 + dx, 0.0), 1.0)
            if wrap and x2 - x1 > 1.0:      # 不許繞超過一整圈
                x1 = x2 - 1.0
        elif hx > 0:
            x2 = x2 + dx if wrap else min(max(x2 + dx, 0.0), 1.0)
            if wrap and x2 - x1 > 1.0:
                x2 = x1 + 1.0
        if hy < 0:
            y1 = min(max(y1 + dy, 0.0), 1.0)
        elif hy > 0:
            y2 = min(max(y2 + dy, 0.0), 1.0)
        det.bbox = [x1, y1, x2, y2]
        self.update()
```

3g. 新框的起點與終點 —— `mousePressEvent` 第 371 行:

```python
        start = _clamp_y_only(self.tf.v2n_point(pos)) if self._wrap() \
            else _clamp01(self.tf.v2n_point(pos))
```

`mouseMoveEvent` 第 397 行:

```python
            end = _clamp_y_only(n) if self._wrap() else _clamp01(n)
```

3h. 存檔語意 —— `mouseReleaseEvent` 第 417 行與 427-428 行的三處 `canonical_bbox` 呼叫都要帶 wrap:

```python
                    pending.bbox = canonical_bbox(pending.bbox, self._wrap())
```

```python
            det = self.selected_det
            if det is not None:
                wrap = self._wrap()
                det.bbox = canonical_bbox(det.bbox, wrap)
                if det.bbox != canonical_bbox(self._orig_bbox, wrap):
                    self.detsEdited.emit()
```

3i. 繪製影像逐圈鋪排 —— `_draw_image`(第 500-517 行)改成:

```python
    def _draw_image(self, painter: QPainter, image_rect: QRectF) -> None:
        assert self._pixmap is not None
        scaled = self._scaled_pixmap() if self.tf.zoom < 1.0 else None
        for k in self._shifts():
            rect = image_rect.translated(k * self.tf.span_x, 0.0)
            if scaled is not None:
                painter.drawPixmap(rect.topLeft(), scaled)
            else:
                visible = rect.intersected(QRectF(self.rect()))
                if visible.isEmpty():
                    continue
                z = self.tf.zoom
                source = QRectF(
                    (visible.left() - rect.left()) / z,
                    (visible.top() - rect.top()) / z,
                    visible.width() / z,
                    visible.height() / z,
                )
                painter.drawPixmap(visible, self._pixmap, source)
        self._draw_image_border(painter, image_rect)

    def _draw_image_border(self, painter: QPainter, image_rect: QRectF) -> None:
        """非環景畫完整外框;環景只畫上下,左右改成接縫參考線。

        環景下左右邊界不存在(那是同一條經線),畫成外框會讓人以為框不能越過去。
        改成一條淡線標出 JSON 的 x=0/1 落在哪 —— 存檔的座標仍以它為原點。
        """
        painter.setPen(QPen(COLOR_IMAGE_EDGE, 1))
        if not self.tf.wrap_x:
            painter.drawRect(image_rect)
            return
        left, right = float(self.rect().left()), float(self.rect().right())
        painter.drawLine(QPointF(left, image_rect.top()), QPointF(right, image_rect.top()))
        painter.drawLine(QPointF(left, image_rect.bottom()),
                         QPointF(right, image_rect.bottom()))
        painter.setPen(QPen(COLOR_SEAM, 1, Qt.PenStyle.DashLine))
        for k in self._shifts():
            x = image_rect.left() + k * self.tf.span_x
            painter.drawLine(QPointF(x, image_rect.top()), QPointF(x, image_rect.bottom()))
```

3i-2. `_scaled_pixmap`(第 519-533 行)不需要改 —— 它只依 `span_x`/`span_y` 產生一張縮圖,每一圈共用同一張。

3j. 繪製框逐圈 —— `_draw_dets`(第 557-569 行)與 `_draw_new_box`(第 595-600 行)改成:

```python
    def _draw_dets(self, painter: QPainter) -> None:
        assert self._frame is not None
        dets = self._frame.dets
        metrics = QFontMetricsF(self._label_font)
        shifts = list(self._shifts())
        for index, det in enumerate(dets):
            base = self.tf.n2v_rect(det.bbox)
            color = det_color(det)
            selected = index == self._sel
            painter.setPen(QPen(color, 3 if selected else 2))
            for k in shifts:
                rect = base.translated(k * self.tf.span_x, 0.0)
                painter.drawRect(rect)
                self._draw_tag(painter, metrics, rect, det.display_text(), color)
        if 0 <= self._sel < len(dets):
            base = self.tf.n2v_rect(dets[self._sel].bbox)
            for k in shifts:
                self._draw_handles(painter, base.translated(k * self.tf.span_x, 0.0))

    def _draw_new_box(self, painter: QPainter) -> None:
        if self._new_det is None:
            return
        painter.setPen(QPen(det_color(self._new_det), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        base = self.tf.n2v_rect(self._new_det.bbox)
        for k in self._shifts():
            painter.drawRect(base.translated(k * self.tf.span_x, 0.0))
```

注意 `_draw_tag` 內部會用 `painter.setPen`,所以框的畫筆要在每次 `drawRect` 前重設。把 `painter.setPen(QPen(color, ...))` 移進 `for k` 迴圈內:

```python
            for k in shifts:
                rect = base.translated(k * self.tf.span_x, 0.0)
                painter.setPen(QPen(color, 3 if selected else 2))
                painter.drawRect(rect)
                self._draw_tag(painter, metrics, rect, det.display_text(), color)
```

3k. `_draw_band`(第 535-555 行)畫的是 y 方向的水平帶,`image_rect.left()/right()` 在環景下只會蓋到主圈。把那三處換成整個 widget 寬度:

```python
    def _draw_band(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._band is None:
            return
        lo, hi = self._band
        y_lo = self.tf.n2v(0.0, lo).y()
        y_hi = self.tf.n2v(0.0, hi).y()
        # 環景下影像左右無限延伸,遮罩與虛線要跨整個視窗寬,否則只蓋住主圈那一份。
        span = QRectF(self.rect()) if self.tf.wrap_x else image_rect

        above = QRectF(span.left(), span.top(), span.width(),
                       max(0.0, y_lo - span.top()))
        below = QRectF(span.left(), y_hi, span.width(),
                       max(0.0, span.bottom() - y_hi))
        for band in (above, below):
            clipped = band.intersected(span)
            if not clipped.isEmpty():
                painter.fillRect(clipped, COLOR_OUTSIDE)

        pen = QPen(COLOR_BAND, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for value, y in ((lo, y_lo), (hi, y_hi)):
            painter.drawLine(QPointF(span.left(), y), QPointF(span.right(), y))
            painter.drawText(QPointF(span.left() + 6.0, y - 4.0), f"y={value:.5f}")
```

`paintEvent` 裡把 `_draw_band` 的呼叫維持原樣(仍傳 `image_rect`),但 `_draw_image` 的無圖分支(第 483-485 行)也要改用新的邊界函式:

```python
        if self._pixmap is None or self._pixmap.isNull():
            self._draw_image_border(painter, image_rect)
            self._draw_hint(painter, "此幀找不到對應影像(frames/ 下無同 stem 檔案)")
```

- [ ] **Step 4: 跑測試 + 手動確認**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling python tests/verify_gui.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 兩支都「全部通過」,含 Step 1b 那個先前 FAIL 的 hit-test 斷言。

再手動開一次確認畫面(這一步不能省 —— 繪製正確性只有眼睛驗得了):

```bash
uv run --project D:\ws\gt_labeling python main.py
```

開 `D:\ws\detect_stream\out\gt_per_frames_0625_182214\160-180s`(掃描當下該段有跨縫框;若已被改動,用計畫開頭的統計腳本另挑一段),確認四件事:

1. 影像左右無縫重複,一直往旁邊拖不會停。
2. 跨縫框畫成**連續的一個矩形**,不是兩截。
3. 在接縫兩側都點得到同一個框,拉手把能跨過接縫。
4. 在接縫上新畫一個框,放開後仍是連續的一個框。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/canvas.py tests/verify_roundtrip.py tests/verify_gui.py
git commit -m "$(cat <<'EOF'
feat(labeling): 畫布左右無縫環繞,跨接縫的框可以直接畫與拖

影像依 visible_shifts 逐圈鋪排,框與 hit-test 同樣逐圈平移,所以接縫兩側
點到的是同一個框。環景下 x 方向不再夾在 [0,1](夾住等於禁止跨縫),y 維持
原樣。左右外框改成接縫參考線,標出 JSON 的 x=0/1 在哪。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 啟用判定、檢視選單與狀態列

**Files:**

- Modify: `gt_labeling/window.py:154-162`(狀態列 widget)、`230-240`(act 定義區)、`289-301`(檢視選單與 addAction 清單)、`470-500`(開資料夾流程)、`1275-1290`(`_refresh_status`)
- Test: `tests/verify_gui.py`(Task 9)+ 手動確認

**Interfaces:**

- Consumes: `FrameLabel.wrap_x` / `is_equirect`(Task 2)、`ImageCanvas.set_wrap_x`(Task 5)
- Produces: `MainWindow.act_wrap: QAction`(checkable)、`MainWindow._apply_wrap_mode(enabled: bool) -> None`

- [ ] **Step 1: 加狀態列欄位**

`window.py` 第 154-162 行加一個 label:

```python
        self.lbl_frame = QLabel("—")
        self.lbl_boxes = QLabel("")
        self.lbl_state = QLabel("")
        self.lbl_wrap = QLabel("")
        self.lbl_zoom = QLabel("")
        self.lbl_pos = QLabel("")
        bar = self.statusBar()
        for widget in (self.lbl_frame, self.lbl_boxes, self.lbl_state,
                       self.lbl_wrap, self.lbl_zoom):
            bar.addWidget(widget)
        bar.addPermanentWidget(self.lbl_pos)
```

- [ ] **Step 2: 定義 action 並掛進檢視選單**

在 `_build_actions` 的 `self.act_fit = QAction("還原檢視", self)`(第 232 行)**之後**加:

```python
        self.act_wrap = QAction("環景模式(ERP)", self)
        self.act_wrap.setCheckable(True)
        # 不給快捷鍵:它改變的是**存檔語意**(x 要不要 clamp 回 [0,1]),
        # 誤按一次就可能把跨縫框削掉,代價遠高於省一次點擊。
        self.act_wrap.toggled.connect(self._apply_wrap_mode)
```

檢視選單(第 289-290 行)改成:

```python
        menu_view = self.menuBar().addMenu("檢視")
        menu_view.addAction(self.act_fit)
        menu_view.addSeparator()
        menu_view.addAction(self.act_wrap)
```

`self.addAction(action)` 的清單(第 295-300 行)**不加** `act_wrap` —— 那個清單是給「快捷鍵不靠選單成立」的動作用的,而這個 action 刻意沒有快捷鍵。

- [ ] **Step 3: 實作模式切換**

在 `_refresh_status`(第 1275 行)**之前**加:

```python
    def _apply_wrap_mode(self, enabled: bool) -> None:
        """切換環景模式:同步每一幀的存檔語意與畫布的檢視語意。

        會讓含跨縫框的幀變成「未存」是正確的 —— 存出去的座標確實會不同。沒有跨縫框
        的幀 dets_json() 不變,所以不會被誤標。
        """
        for frame in self.frames:
            frame.wrap_x = enabled
        self.canvas.set_wrap_x(enabled)
        self.list_panel.set_frames(self.frames)
        if 0 <= self.index < len(self.frames):
            self.list_panel.set_current(self.index)
        self._refresh_status()
```

- [ ] **Step 4: 開資料夾時自動判定**

在 `open_root` 裡,`self.list_panel.set_frames(frames)`(第 491 行)**之前**插入:

```python
        # 2:1 就進環景模式。JSON 沒有、也不會加「這是 equirect」的欄位,size 是唯一
        # 線索;setChecked 會觸發 toggled 走 _apply_wrap_mode,把 frames 與畫布一起設好。
        wrap = bool(frames) and frames[0].is_equirect
        if self.act_wrap.isChecked() == wrap:
            self._apply_wrap_mode(wrap)      # 值沒變不會發 toggled,手動補一次
        else:
            self.act_wrap.setChecked(wrap)
```

注意 `load_frame` 已經把每個 frame 的 `wrap_x` 設好(Task 2),這裡是為了讓 UI 狀態與之對齊、並處理使用者換資料夾的情況。

- [ ] **Step 5: 狀態列顯示模式**

`_refresh_status`(第 1275-1285 行)加兩行:

```python
    def _refresh_status(self) -> None:
        if not (0 <= self.index < len(self.frames)):
            self.lbl_frame.setText("未開啟資料")
            for label in (self.lbl_boxes, self.lbl_state, self.lbl_wrap,
                          self.lbl_zoom, self.lbl_pos):
                label.setText("")
            return
        frame = self.frames[self.index]
        self.lbl_frame.setText(f"{self.index + 1}/{len(self.frames)}   seq={frame.seq}")
        self.lbl_boxes.setText(f"  框 {len(frame.dets)}  待補 {frame.pending_count}")
        self.lbl_state.setText("  未存 *" if frame.dirty else "  已存")
        self.lbl_wrap.setText("  ERP 環景" if frame.wrap_x else "")
        self.lbl_zoom.setText(f"  zoom {self.canvas.tf.zoom * 100:.0f}%")
```

`_refresh_status` 第 1286 行之後若還有其他內容(游標位置等)維持原樣不動。

- [ ] **Step 6: 跑測試 + 手動確認**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
uv run --project D:\ws\gt_labeling python main.py
```

手動確認:

1. 開 `...\gt_per_frames_0625_182214\160-180s` → 檢視選單的「環景模式(ERP)」自動打勾,狀態列出現「ERP 環景」。
2. 取消勾選 → 畫布回到單張、狀態列的字消失、含跨縫框的幀在清單上出現 `*`(未存)。
3. 重新勾選 → 那些 `*` 消失(回到與磁碟一致)。

- [ ] **Step 7: Commit**

```bash
git add gt_labeling/window.py
git commit -m "$(cat <<'EOF'
feat(labeling): 2:1 影像開檔自動進環景模式,檢視選單可手動覆寫

size 是判定 equirect 的唯一線索(JSON 沒有這個欄位)。刻意不給快捷鍵:
它改變的是存檔語意,誤按就可能削掉跨縫框。狀態列顯示目前模式。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: 幀清單的 `CUT` 貼邊警示

**Files:**

- Modify: `gt_labeling/panel.py:33-35`(顏色)、`45-46`(docstring)、`79`(標題列)、`93-127`(`refresh_row` / `refresh_summary`)
- Test: `tests/verify_roundtrip.py`(model 層)+ `tests/verify_gui.py`(Task 9)

**Interfaces:**

- Consumes: `FrameLabel.has_edge_box`(Task 2 已實作)
- Produces: 無新 API,只有 UI 呈現

- [ ] **Step 1: 寫失敗測試**

在 `tests/verify_roundtrip.py` 的 `test_frame_wrap_state()` 尾端追加:

```python
    cut = frame_with([3840, 1920], [[0.94063, 0.5, 1.0, 0.7]])
    check(cut.has_edge_box, "x2 恰好 1.0 → 判定為被削過的貼邊框")
    left_cut = frame_with([3840, 1920], [[0.0, 0.5, 0.0356, 0.7]])
    check(left_cut.has_edge_box, "x1 恰好 0.0 → 同樣判定為貼邊")
    healthy = frame_with([3840, 1920], [[0.94063, 0.5, 1.00348, 0.7]])
    check(not healthy.has_edge_box, "x2 > 1 是健康的跨縫框,不算貼邊")
    inner = frame_with([3840, 1920], [[0.3, 0.4, 0.5, 0.6]])
    check(not inner.has_edge_box, "畫面內的框不算貼邊")
    flat_cut = frame_with([1920, 1080], [[0.94063, 0.5, 1.0, 0.7]])
    check(not flat_cut.has_edge_box, "非 equirect 資料貼邊是正常的,不警示")
```

- [ ] **Step 2: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: 全部通過(`has_edge_box` 在 Task 2 已實作)。若 FAIL 表示 Task 2 的實作與這裡的預期不符,先修 model 再往下。

- [ ] **Step 3: 實作 UI**

3a. `panel.py` 第 33-35 行加顏色:

```python
COLOR_PENDING = QColor("#ffb340")
COLOR_DUPLICATE = QColor("#ff5f56")
COLOR_EDGE = QColor("#ffd24a")
COLOR_NORMAL = QColor("#dcdcdc")
```

3b. `FrameListPanel` 的 docstring(第 46 行)改成:

```python
    """每幀一列:seq、框數、待補標記(ID / PPE)、id 重疊(DUP)、貼邊(CUT)、未存(*)。"""
```

3c. 標題列(第 79 行)改成:

```python
        layout.addWidget(QLabel("幀清單  seq  框數  待補  重疊 貼邊 未存"))
```

3d. `refresh_row`(第 93-114 行)改成:

```python
    def refresh_row(self, index: int, frame: FrameLabel) -> None:
        item = self.list.item(index)
        if item is None:
            return
        flag_id = "ID " if frame.has_null_track else "   "
        flag_ppe = "PPE" if frame.has_null_ppe else "   "
        duplicate = frame.has_duplicate_track
        edge = frame.has_edge_box
        flag_dup = "DUP" if duplicate else "   "
        flag_cut = "CUT" if edge else "   "
        dirty = "*" if frame.dirty else " "
        item.setText(
            f"{frame.seq:>7d}  n={len(frame.dets):<3d} "
            f"{flag_id}{flag_ppe} {flag_dup} {flag_cut}  {dirty}"
        )

        # 三級:重疊是已經填錯(最重),貼邊是框可能被 clamp 削過、要回頭看一眼,
        # 待補只是還沒填。顏色依此壓過去。
        pending = frame.has_null_track or frame.has_null_ppe
        if duplicate:
            item.setForeground(COLOR_DUPLICATE)
        elif edge:
            item.setForeground(COLOR_EDGE)
        else:
            item.setForeground(COLOR_PENDING if pending else COLOR_NORMAL)
        font = _mono_font()
        font.setBold(frame.dirty)
        item.setFont(font)
```

3e. `refresh_summary`(第 116-127 行)加一行計數:

```python
    def refresh_summary(self, frames: list[FrameLabel]) -> None:
        boxes = sum(len(f.dets) for f in frames)
        pending_boxes = sum(f.pending_count for f in frames)
        pending_frames = sum(1 for f in frames if f.pending_count)
        dirty_frames = sum(1 for f in frames if f.dirty)
        duplicate_frames = sum(1 for f in frames if f.has_duplicate_track)
        edge_frames = sum(1 for f in frames if f.has_edge_box)
        self.summary.setText(
            f"{len(frames)} 幀 / {boxes} 框\n"
            f"待補 {pending_boxes} 框(分布 {pending_frames} 幀)\n"
            f"id 重疊 {duplicate_frames} 幀\n"
            f"貼邊 {edge_frames} 幀\n"
            f"未存 {dirty_frames} 幀"
        )
```

- [ ] **Step 4: 跑測試 + 手動確認**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
uv run --project D:\ws\gt_labeling python main.py
```

手動:開 `...\gt_per_frames_0625_182214\100-120s`(那段有被削過的框),確認清單出現黃色 `CUT` 列、摘要「貼邊 N 幀」的 N > 0;開 `160-180s`(跨縫框健康)確認 N = 0。

- [ ] **Step 5: Commit**

```bash
git add gt_labeling/panel.py tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
feat(labeling): 幀清單標出貼邊框,找得到先前被 clamp 削掉尾巴的框

環景下框恰好貼在 x=0 或 x=1 幾乎一定是被削過的痕跡(x2>1 才是健康的跨縫
框)。這同時是回溯手段與往後的防呆:任何原因造成的貼邊都會被看見。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 真實資料回歸 —— 上游跨縫框存回去必須 byte 相同

**Files:**

- Modify: `tests/verify_roundtrip.py`(新增 `test_wrap_real_data`、`main()` 的參數處理)
- Test: 同上

**Interfaces:**

- Consumes: Task 1-3 的 model 層行為
- Produces: `test_wrap_real_data(sample: Path) -> None`

這是整個計畫最有力的一項:拿上游真的產出的跨縫框跑一輪 load→save,**現行程式必失敗、改完必通過**。

- [ ] **Step 1: 寫測試**

在 `tests/verify_roundtrip.py` 的 `test_edit_roundtrip` **之後**新增:

```python
def test_wrap_real_data(sample: Path) -> None:
    """上游 gt_densify 產出的跨縫框(x2>1),讀進來再存回去必須 byte 完全相同。

    這是本次改動的核心回歸:改動前 canonical_bbox 會把 x2 夾到 1.0,那些框每存一次
    就少一截,而且沒有任何流程會報錯。

    不斷言跨縫框的個數 —— 標註正在進行中,數量隨時在變。只斷言「有跨縫框」與
    「全數不變」這兩件與資料量無關的性質。
    """
    section(f"真實資料回歸:上游跨縫框 round-trip({sample.name})")
    entries = scan_root(sample)
    crossing = 0
    identical = 0
    checked = 0
    for entry in entries:
        before = entry.label_path.read_bytes()
        frame = load_frame(entry.label_path)
        crossing += sum(1 for d in frame.dets if d.bbox[2] > 1.0 or d.bbox[0] < 0.0)
        was_dirty = frame.dirty
        frame.save(force=True)
        checked += 1
        if entry.label_path.read_bytes() == before:
            identical += 1
        if was_dirty:
            print(f"       注意:{entry.label_path.name} 一讀進來就被標成已改")
    check(crossing > 0,
          f"這份樣本含跨縫框可供回歸(實際 {crossing} 個);為 0 請換一段有人走過接縫的資料")
    check(identical == checked,
          f"強制重寫後 byte 完全相同:{identical}/{checked} 檔")
```

- [ ] **Step 2: 讓 `main()` 吃第二個樣本路徑**

把 `main()`(第 319-350 行)改成:

```python
def main() -> int:
    source = Path(
        sys.argv[1] if len(sys.argv) > 1
        else r"D:\ws\detect_stream\out\gt_per_frames_0625_145125\000-020s"
    )
    if not (source / "labels").is_dir():
        print(f"找不到 {source}\\labels")
        return 2

    # 第二份樣本專供跨縫回歸:必須是「有人走過接縫」的段落。預設指向已知有跨縫框
    # 的一段;路徑不存在就跳過該項而不是 FAIL —— 資料集會搬、會被重新切段,拿
    # 找不到檔案當失敗只會製造與程式無關的假警報。
    wrap_sample = Path(
        sys.argv[2] if len(sys.argv) > 2
        else r"D:\ws\detect_stream\out\gt_per_frames_0625_182214\160-180s"
    )

    with tempfile.TemporaryDirectory(prefix="gt_verify_") as tmp:
        work = Path(tmp) / "gt_sample"
        # 只複製 labels(影像不需要),frames 建空目錄讓 scan_root 通過。
        (work / "frames").mkdir(parents=True)
        shutil.copytree(source / "labels", work / "labels")
        print(f"工作副本:{work}")

        test_unchanged_save_is_byte_identical(work)
        test_preserves_unknown_det_fields(work)
        test_edit_roundtrip(work)

    if (wrap_sample / "labels").is_dir():
        with tempfile.TemporaryDirectory(prefix="gt_wrap_") as tmp:
            work = Path(tmp) / "gt_wrap"
            (work / "frames").mkdir(parents=True)
            shutil.copytree(wrap_sample / "labels", work / "labels")
            test_wrap_real_data(work)
    else:
        section("真實資料回歸:上游跨縫框 round-trip")
        print(f"  skip 找不到 {wrap_sample}\\labels,跳過(可用第二個參數指定)")

    test_interpolate_is_per_label()
    test_interpolate_wraps_shortest_arc()
    test_canonical_bbox()
    test_frame_wrap_state()
    test_transform_roundtrip()
    test_transform_wrap()
    test_canvas_wrap_geometry()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失敗 {len(FAILURES)} 項:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("全部通過")
    return 0
```

- [ ] **Step 3: 用 `git stash` 確認這個測試抓得到 bug**

```bash
git stash push gt_labeling/model.py
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: `FAIL 強制重寫後 byte 完全相同` —— 舊的 `canonical_bbox` 會把 172 個跨縫框削平,檔案內容改變。**這一步證明測試有效**,沒看到 FAIL 表示測試沒抓到東西,要回頭查。

```bash
git stash pop
```

- [ ] **Step 4: 跑測試確認通過**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 「全部通過」,含 `這份樣本含跨縫框可供回歸(實際 172 個)`。

- [ ] **Step 5: Commit**

```bash
git add tests/verify_roundtrip.py
git commit -m "$(cat <<'EOF'
test(labeling): 加上游跨縫框的真實資料回歸,改動前必失敗

拿 gt_densify 真的產出的 x2>1 的框跑 load→save,斷言 byte 完全相同。
不寫死跨縫框個數 —— 標註進行中,數量隨時在變,只斷言「有」與「全數不變」。
樣本路徑找不到就跳過而非 FAIL。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: GUI 互動驗收、下游相容性檢查與 README

**Files:**

- Modify: `tests/verify_gui.py`(新增環景區段)、`README.md`
- Test: `tests/verify_gui.py`

**Interfaces:**

- Consumes: Task 1-7 的全部行為
- Produces: 無新 API

- [ ] **Step 1: GUI 環景區段**

在 `tests/verify_gui.py` 的 `main()` 裡,`section("開資料夾")` 那一段的檢查之後(第 194 行 `"首幀尺寸 3840x1920"` 之後)插入:

```python
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

        # 貼邊警示:造一個 x2 恰好 1.0 的框,幀清單那一列要出現 CUT
        window._goto(0)
        app.processEvents()
        frame0 = window.frames[0]
        frame0.dets.append(
            Det(label="person", track_id=778, ppe="ng", bbox=[0.96, 0.30, 1.0, 0.45])
        )
        window.list_panel.refresh_row(0, frame0)
        window.list_panel.refresh_summary(window.frames)
        app.processEvents()
        check(frame0.has_edge_box, "x2 恰好 1.0 → model 判定為貼邊")
        check("CUT" in window.list_panel.list.item(0).text(),
              f"幀清單那一列出現 CUT 旗標 {window.list_panel.list.item(0).text()!r}")
        check("貼邊 1 幀" in window.list_panel.summary.text(),
              f"摘要計數正確 {window.list_panel.summary.text()!r}")
        del frame0.dets[-1]
        window.list_panel.refresh_row(0, frame0)
        window.list_panel.refresh_summary(window.frames)
        check("CUT" not in window.list_panel.list.item(0).text(), "移除後 CUT 消失")

        # 關掉環景:含跨縫框的幀變成未存(存出去確實會不同)
        window.act_wrap.setChecked(False)
        app.processEvents()
        check(not canvas.tf.wrap_x, "取消勾選後畫布回到非環景")
        check(not window.frames[0].wrap_x, "model 的存檔語意也跟著回去")
        check(window.lbl_wrap.text() == "", "狀態列的環景字樣消失")
        window.act_wrap.setChecked(True)
        app.processEvents()
        check(canvas.tf.wrap_x, "重新勾選回到環景")
```

`QPoint` / `QTest` / `Qt` 已在 verify_gui.py 匯入;`Det` 若尚未匯入,在 `from gt_labeling.model import ...` 那行補上(Task 5 Step 1b 可能已經補過)。

- [ ] **Step 2: 跑 GUI 驗收**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_gui.py
```

Expected: 「全部通過」。兩個排查方向,都**不要**用調寬容忍值的方式繞過:

- `接縫上畫出一個新框` FAIL 且新框數沒變 → 起點可能命中了既有的框(那會變成 MOVE 而不是 NEW)。印出 `canvas.selected_index`:非 -1 就是命中了,換一個 y 座標(例如 `500` / `560`)再試。
- 新框數沒變且 `selected_index` 是 -1 → `seam_x` 可能落在 canvas 之外。印出 `seam_x`、`canvas.width()`、`canvas.tf.span_x`、`canvas.tf.off_x` 四個值對照。

- [ ] **Step 3: 下游相容性檢查**

在 `tests/verify_roundtrip.py` 的 `test_canvas_wrap_geometry()` **之後**新增:

```python
def test_downstream_wrap_iou() -> None:
    """延伸表示能被下游的環狀 IoU 正確配對。

    detect_stream 的 evaluate.py / eval_gt.py 用 wrap_iou(x 平移 ±1 取最大)吃跨縫框。
    這裡**只讀不寫**那個 repo,驗證我們落檔的表示它接得住;找不到就跳過,不讓本 repo
    的測試硬綁另一個 repo 的位置。
    """
    section("下游相容:wrap_iou 吃得下延伸表示")
    eval_dir = Path(r"D:\ws\detect_stream\scripts\eval2")
    if not (eval_dir / "evaluate.py").is_file():
        print(f"  skip 找不到 {eval_dir}\\evaluate.py,跳過")
        return
    sys.path.insert(0, str(eval_dir))
    try:
        import numpy as np
        from evaluate import wrap_iou
    except ImportError as exc:
        print(f"  skip 匯入失敗({exc}),跳過")
        return
    finally:
        sys.path.remove(str(eval_dir))

    extended = np.asarray([[0.94063, 0.5, 1.00348, 0.7]], dtype=float)
    same = np.asarray([[0.94063, 0.5, 1.00348, 0.7]], dtype=float)
    check(round(float(wrap_iou(extended, same)[0, 0]), 6) == 1.0,
          "同一個延伸表示的框互比 IoU = 1.0")

    # 對側等價寫法(整體平移一圈)也該配上
    shifted = np.asarray([[-0.05937, 0.5, 0.00348, 0.7]], dtype=float)
    check(round(float(wrap_iou(extended, shifted)[0, 0]), 6) == 1.0,
          "平移一整圈的等價框 IoU = 1.0(wrap_iou 的 ±1 平移生效)")

    # x2 < x1 那種寫法會被算成零面積 —— 這正是我們不採用它的理由,鎖成回歸
    reversed_form = np.asarray([[0.94063, 0.5, 0.00348, 0.7]], dtype=float)
    check(float(wrap_iou(reversed_form, same)[0, 0]) == 0.0,
          "x2<x1 的寫法在 wrap_iou 下是零面積(所以本工具不採用)")
```

在 `main()` 的 `test_canvas_wrap_geometry()` 呼叫之後加一行:

```python
    test_downstream_wrap_iou()
```

跑一次:

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
```

Expected: 全部通過,或印出 `skip`(detect_stream 不在預期位置時)。注意 detect_stream 需要 numpy,若本 repo 的 venv 沒有 numpy 就會走 skip 分支 —— 那是預期行為,**不要**為此把 numpy 加進 `pyproject.toml` 的 dependencies。

- [ ] **Step 4: 更新 README**

4a. 第 80 行 `bbox` 的說明改成:

```markdown
- `bbox`:normalized `[x1, y1, x2, y2]`,**這是唯一真值**,畫面像素只是投影。equirect(size 寬高比 2:1)的跨接縫框以 `x1 ∈ [0,1)` + `x2 = x1 + 寬度`(可 > 1)表示,見下方「環景模式」
```

4b. 在「操作」章節(第 83 行起)裡新增一節:

```markdown
### 環景模式(ERP)

`size` 寬高比 2:1 的資料開檔就自動進環景模式,狀態列顯示「ERP 環景」,檢視選單可手動覆寫。

- 畫布左右**無縫環繞**:一直往旁邊拖不會停,接縫可以拉到畫面中間再畫框。淡藍虛線標出 JSON 的 `x=0/1` 落在哪。
- 跨接縫的框畫成**連續的一個矩形**,接縫兩側都點得到同一個框。
- 落檔時 x 不 clamp:`x1` 取模回 `[0,1)`、`x2` 寫成 `x1 + 寬度`(可越過 1.0)。這是上游 `gt_densify.py` 產出、下游 `eval_gt.py` / `evaluate.py` 的 `wrap_iou` 消費的既有表示 —— 夾回 `[0,1]` 會把跨縫的人切成半個框,而且不會有任何流程報錯。
- `x2 < x1` 那種寫法**不能用**:`wrap_iou` 會算成負寬、面積 0,該框在評估裡永遠是 FN。
- y 不環繞:equirect 的上下是極點,越過去取到的是另一側的天空/地面。
- 補框在環景下走 ERP 最短弧,走過接縫的人不會補出一串橫掃整張圖的假框。
- 幀清單的 `CUT` 標出「有框恰好貼在 x=0 或 x=1」—— 那幾乎一定是先前被 clamp 削過的痕跡,`x2 > 1` 才是健康的跨縫框。
- 手動關掉環景模式後存檔**會把跨縫框 clamp 掉**,含跨縫框的幀會立刻變成未存(`*`),因為存出去的座標確實會不同。
```

4c. 存檔契約第 4 條(第 260 行)改成:

```markdown
4. `bbox` 一律寫成 5 位小數、保證 `x1<x2` / `y1<y2`。`y` 一律 clamp 到 `[0,1]`;`x` 在環景模式下**不** clamp,以 `x1 ∈ [0,1)` + `x2 = x1 + 寬度` 落檔,`x2` 可越界。非環景模式下 `x` 維持 clamp 到 `[0,1]`。
```

4d. 「驗收」章節補上第二個樣本參數的說明:

```markdown
`tests/verify_roundtrip.py` 吃兩個可選路徑參數:第一個是一般樣本,第二個是**含跨縫框**的樣本(預設 `gt_per_frames_0625_182214\160-180s`)。第二個找不到就跳過該項而不是 FAIL —— 資料集會搬、會被重新切段,拿找不到檔案當失敗只會製造假警報。
```

- [ ] **Step 5: 全套驗收**

```bash
uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py
uv run --project D:\ws\gt_labeling python tests/verify_gui.py
uv run --project D:\ws\gt_labeling ruff check .
```

Expected: 兩支都「全部通過」,ruff 無錯。

- [ ] **Step 6: Commit**

```bash
git add tests/verify_gui.py tests/verify_roundtrip.py README.md
git commit -m "$(cat <<'EOF'
test(labeling): 補環景畫布的 GUI 驗收與下游 wrap_iou 相容檢查

GUI 驗收涵蓋自動判定、水平無限 pan、接縫上新畫框、切換模式。相容檢查
只讀 detect_stream 的 wrap_iou,順便把「x2<x1 是零面積」鎖成回歸,說明
為什麼不採用那種寫法。README 補環景模式一節與存檔契約第 4 條。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 收尾檢查

全部 task 完成後,對照 spec 逐項確認:

- [ ] 開 `...\gt_per_frames_0625_182214\160-180s`,跨縫框畫成連續矩形、可拖可縮放
- [ ] 開 `...\100-120s`,幀清單出現 `CUT`,把其中一個貼邊框的尾巴拉回來、存檔、重開,座標保持
- [ ] `git diff main --stat` 確認**沒有**任何 `D:\ws\detect_stream` 下的檔案被改到
- [ ] `uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py` 全綠
- [ ] **換一份沒有跨縫框的 equirect 資料再跑一次**,確認「進了環景模式但輸出逐位元不變」:
      `uv run --project D:\ws\gt_labeling python tests/verify_roundtrip.py D:\ws\detect_stream\out\gt_per_frames_0626_135335\000-020s`
      (掃描當下該資料集 0 個跨縫框、0 個貼邊,是驗這件事的乾淨對照組)
- [ ] `uv run --project D:\ws\gt_labeling python tests/verify_gui.py` 全綠
- [ ] `uv run --project D:\ws\gt_labeling ruff check .` 無錯
