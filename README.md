# gt-labeling

GT 標註修正工具:檢視與修正物件偵測 ground truth 的 PyQt6 桌面程式。

針對「系統輸出的偵測結果已經有八成對,只需要人工修掉錯的、補上漏的」這個場景設計 —— 不是從零開始標註,而是**修正既有 GT**。因此整個工具的正確性核心只有一條:**存回去的 JSON 除了你真的改過的 `dets`,其餘一個 byte 都不能動**。

## 需求

- Python >= 3.11
- PyQt6 >= 6.7

## 安裝與啟動

用 uv(不必手動啟用 venv):

```
uv run --project D:\ws\gt_labeling python main.py D:\ws\detect_stream\out\gt_per_frames_0625_145125\000-020s
```

或啟用 venv 後執行:

```
D:\ws\gt_labeling> .venv\Scripts\activate
(gt-labeling) D:\ws\gt_labeling> python main.py D:\ws\detect_stream\out\gt_per_frames_0625_145125\000-020s
```

安裝成套件後也可以直接用 console script:

```
uv run --project D:\ws\gt_labeling gt-labeling <root> --band 0.5 0.9
```

### 命令列參數

| 參數           | 說明                                                          |
| -------------- | ------------------------------------------------------------- |
| `root`         | 含 `frames/` 與 `labels/` 的資料夾;省略則開啟資料夾選擇對話框 |
| `--band Y1 Y2` | 偵測帶上下界(歸一化 y,0~1),畫成水平參考線,帶外壓暗提醒不要標  |
| `--cache N`    | 解碼後 QPixmap 的快取張數(預設 8)                             |

## 資料夾結構

```
<root>/
  frames/     影像:.jpg .jpeg .png .bmp .webp .tif .tiff
  labels/     標註:.json
```

`frames/` 與 `labels/` **以檔名 stem 對應**(`000123.jpg` ↔ `000123.json`)。JSON 裡的 `image` 欄位只原樣保留、不參與定位 —— 它是 `../frames/...` 相對路徑,拿它定位會綁死目錄結構。

檔名是純數字就直接當 `seq`(最省),否則退回讀 JSON 的 `seq` 欄位。影像缺檔不影響開啟,標題列會顯示缺了幾張。

## 標註檔格式

```json
{
  "type": "...",
  "version": "...",
  "source": "...",
  "seq": 123,
  "frame_index": 123,
  "video_sec": 4.1,
  "size": [3840, 1920],
  "image": "../frames/000123.jpg",
  "dets": [
    {
      "label": "person",
      "track_id": 1,
      "ppe": "ng",
      "bbox": [0.3125, 0.441, 0.359, 0.612],
      "src": "det"
    }
  ]
}
```

- `label`:`person` 或 `drone`
- `track_id`:整數或 `null`(待指定)
- `ppe`:`ok` / `ng` / `null`,只對 `person` 有意義;`drone` 一律寫 `null`
- `bbox`:normalized `[x1, y1, x2, y2]`,**這是唯一真值**,畫面像素只是投影
- `size`:必要欄位,缺了無法換算 normalized 座標,會直接報錯

## 操作

### 鍵盤

| 快捷鍵                    | 動作                                      |
| ------------------------- | ----------------------------------------- |
| `A` / `←` / `PageUp`      | 上一張                                    |
| `D` / `→` / `PageDown`    | 下一張                                    |
| `Home` / `End`            | 第一幀 / 最後一幀                         |
| `Delete` / `Backspace`    | 焦點在畫布:刪除選取的框                   |
| `Delete` / `Backspace`    | 焦點在幀清單:刪掉選取幀內的整條 track     |
| `F`                       | 還原檢視(fit)                             |
| `Esc`                     | 取消進行中的拖曳                          |
| `Ctrl+O`                  | 開資料夾                                  |
| `Ctrl+S`                  | 存檔                                      |
| `Ctrl+Z`                  | 復原                                      |
| `Ctrl+Shift+Z` / `Ctrl+Y` | 重做                                      |
| `Ctrl+I`                  | 內插補框(選取框所屬的 track)              |
| `Ctrl+R`                  | 改 id(選取框所屬 track 全域換號)          |
| `Ctrl+Shift+I`            | 整組復原上一次批次(補框 / 改 id / 刪軌跡) |
| `Ctrl+F`                  | 聚焦 track 搜尋欄(預填選取框)             |
| `F3` / `Shift+F3`         | 找下一個 / 上一個出現該 track 的框        |

搜尋欄內另有:`Enter` 下一個、`Shift+Enter` 上一個、`Esc` 把焦點還給畫布。

### 滑鼠

| 操作                           | 動作                         |
| ------------------------------ | ---------------------------- |
| 滾輪                           | 以游標為錨點縮放             |
| 中鍵 / 右鍵 / `Space`+左鍵拖曳 | 平移                         |
| 左鍵拖空白處                   | 新增框(套用右側「新框預設」) |
| 左鍵點框                       | 選取                         |
| 拖框內                         | 整體移動(不變形)             |
| 拖角/邊控制點                  | 改大小                       |

### 框的顏色

| 顏色 | 意義           |
| ---- | -------------- |
| 綠   | person, ppe=ok |
| 橘   | person, ppe=ng |
| 紅   | drone          |

## 功能

### 新框預設

右上「新框預設」決定接下來畫的框帶什麼屬性,不影響已存在的框:

- **label**:person(自動判 `ppe=ng`)或 drone
- **track_id**:
  - `自動`:取**本幀**已用號碼的最大值 +1(不是取陣列最後一顆 —— tracker 輸出順序不保證遞增)
  - `沿用 #id`:先點該 track 任一個框,之後畫的新框都掛同一個 id。補錨點連畫好幾幀時用這個。候選號碼跟著「最後碰過的那條軌跡」走 —— 在右側改掉選取框的 `track_id`、或整條軌跡換號(`Ctrl+R`)後都會換成新號,不會停在剛淘汰的舊號

### 內插補框

逐幀補漏標很慢。改成:只在稀疏的幾幀手動畫「錨點」,再讓工具把中間補滿。

用法:右側勾選「啟用內插補框」→ 點選目標 track 的任一個框 → `Ctrl+I` → 確認。

規則:

- 軌跡身分是 **`(label, track_id)`** 而不是單獨的 `track_id`。下游 `eval_gt.py` 把 person / drone 拆成兩個清單各自評估,`gt_densify.py --drone-id` 也只保證同一架 drone 統一成一個號、不保證跟 person 不撞。只認 track_id 的話 person#1 與 drone#1 會被當成同一條軌跡,輕則洞被對方的框填掉而靜默不補,重則內插出一個地面與天花板中點的捏造框。
- 只在**錨點間距 <= 門檻**時補。這道門檻同時擋兩件事:內插誤差(實測 20 幀間距 IoU 中位 0.78、失準率 0.2%,30 幀就跳到 9.3%),以及**遮擋** —— 目標被擋住的區段人不會去標錨點,間距自然拉大而被拒絕,不會憑空生出錯誤的框。
- 建議門檻:drone 20 幀、person 30 幀(預設 20)。
- 權重用 `seq` 差而非清單位置差,抽樣不連續的資料集也不會算歪。
- 被跳過的洞會在對話框列出,並說明該補哪裡。
- `Ctrl+Shift+I` 把上一次補框的所有幀整組還原(與「改 id」共用同一個還原點)。

### 斷軌換號(改 id)

tracker 斷軌時同一個目標會被切成兩個號碼,逐幀改號很慢。改成:點該段任一個框,一次把整條軌跡換掉。

用法:點選要改的那段任一個框 → `Ctrl+R` → 輸入目標 `track_id` → 確認。

規則:

- 軌跡身分同樣是 **`(label, track_id)`**。把 person#7 改成 #3 不會動到 drone#7 —— 理由與內插補框相同:下游把 person / drone 拆成兩個清單各自評估,串號不會讓任何流程報錯,只會靜默算錯。
- 目標號碼**已存在於某些幀**時先警告並列出那些 seq:改完那幾幀會同時出現兩個同號框。斷軌的兩段通常各佔不同的幀、不會重疊,所以重疊多半代表這兩段其實不是同一個目標 —— 看過再決定要不要照做。
- 只有目標號、沒有來源號的幀不算衝突 —— 那正是斷軌另一段本來就該保留的框。
- 只改 `track_id`,不動 bbox / ppe / label,不新增也不刪除框。
- `Ctrl+Shift+I` 整組復原。這個還原點與補框共用,**後做的那次會蓋掉前一次**;單幀 `Ctrl+Z` 只救得回一幀。

### 刪掉一段軌跡

tracker 常在目標離場後還吐一串幽靈框,或某一段被誤配到別的目標。這種錯誤總是「一整段」,逐幀點框刪在幾百幀的序列上慢到沒人會做。改成:圈出範圍,一次刪完。

用法:點該 track 任一個框(記住要刪哪一條)→ 到左側幀清單用 `Shift` 拉連續範圍(或 `Ctrl` 加點零散的幀)→ `Delete` → 確認。

規則:

- 要刪哪一條,取自**最後點過的那個框**。做成黏著值是因為切幀會清掉畫布選取,而這個流程中間必然切過幀 —— 按下 `Delete` 時已經問不到「剛剛選的是誰」。確認視窗一定寫出 `label #id`,不會默默刪錯條。
- 軌跡身分同樣是 **`(label, track_id)`**:刪 person#7 不會動到同幀的 drone#7,理由同內插補框與斷軌換號。
- 只刪**選取範圍內**的框,範圍外的同一條軌跡原封不動。確認視窗會寫出範圍外還剩幾個框 —— 剩 0 代表這條軌跡會從整份資料集消失,後果差很多。
- 同一幀有多個同號框(清單的 `DUP`)時**每個都刪**,不會只刪一個讓人以為清乾淨了。
- 焦點在幀清單時 `Delete` 才是這個動作;焦點在畫布時仍是「刪掉選取的那一個框」。人在清單裡拉範圍時焦點不會被搶走,所以 `↑↓` 與 `Delete` 一路有效;刪完焦點交還畫布,`A` / `D` 翻頁跟著回來。
- `Ctrl+Shift+I` 整組復原。這個還原點與補框、改 id 共用,**後做的那次會蓋掉前一次**。

### 找 track

追一條軌跡時要一直翻幀找「它下次出現在哪」。改成:輸入號碼直接跳過去。

用法:`Ctrl+F`(有選取框就把它的 `label` 與 `track_id` 一起預填)→ `Enter` 連按走完整條軌跡。焦點在畫布時用 `F3` / `Shift+F3`,不必回到輸入框。

規則:

- 搜尋條件是 **`(label, track_id)`**,理由同內插補框與斷軌換號。label 下拉留「全部」則不分 label,用在「不確定號碼掛在哪一種」時掃一遍。
- 跳過去會**選取**那個框,所以可以接著按 `Ctrl+I` 補框、`Ctrl+R` 改 id,或直接拖曳修正。
- 同一幀有兩個同號框(清單的 `DUP`)時**逐個停**,不會整幀跳過 —— 要修的正是第二個。
- 起點含目前選取的框:沒選框時當前幀的第一個命中會先被選起來,不會直接跳走。
- 掃到盡頭繞回另一端,訊息會註明。狀態列同時報「第幾次 / 共幾次出現」,可以當成這條軌跡的長度速查。
- 找不到會**彈警示視窗**擋下來,不只在狀態列閃一下。若是限定了 label 才落空,視窗會直接說出這個號碼其實掛在哪一種 label —— person 與 drone 常各自從 0 開始編號,下拉選錯是最常見的原因。
- 純查詢,不動任何資料。

### 偵測帶

`--band Y1 Y2` 或右側面板設定歸一化 y 的上下界,畫成水平參考線,帶外壓暗 —— 提醒那個區域本來就不該有偵測、不要標。

### 待補追蹤

左側幀清單每列顯示 `seq / 框數 / 待補標記 / 重疊標記 / 未存標記`:

- `ID`:該幀有 `track_id` 為 null 的框
- `PPE`:該幀有 person 的 `ppe` 未判定
- `DUP`:該幀有兩個框共用同一個 `(label, track_id)`
- `*`:該幀有未存檔的修改(整列變粗體)
- 有待補的整列變橘色,有 id 重疊的整列變紅色(壓過橘色 —— 待補是還沒填,重疊是已經填錯)

`DUP` 只認同一個 `(label, track_id)`。person#1 與 drone#1 同幀是正常的、不算重疊 —— 兩種 label 常各自從 0 開始編號,算進來的話警示會每幀都亮,等於沒有警示。換號撞到既有號碼(見「斷軌換號」)、或上游把兩個目標配成同一條軌跡,都會在這裡亮出來。

底下的統計摘要顯示全資料集的待補框數、分布幀數、id 重疊幀數與未存幀數。

### 存檔行為

- 「切幀自動存」預設開啟:切換幀時自動寫回有改動的幀。
- 關閉時改成跳詢問(存 / 丟棄 / 取消);未存的編輯會留在記憶體,可跨幀保留。
- 關閉程式或切換資料夾時會統一處理所有未存的幀。
- 寫檔用「寫 `.tmp` 再 `os.replace`」,不會出現寫到一半的半截檔。

## 存檔契約

這是本工具的驗收依據,改動 [gt_labeling/model.py](gt_labeling/model.py) 時必須維持:

1. **只替換 `raw["dets"]`**。`type / version / source / seq / frame_index / video_sec / size / image` 連 key 順序都是原本那顆 dict,不重建。
2. **det 也一樣**:只覆寫 `label / track_id / ppe / bbox`,其餘欄位連 key 順序原樣帶回。上游 `gt_densify.py` 會寫 `src`(anchor / det / interp),`eval_gt.py` 靠它算「多少比例的框取自系統輸出」—— 剝掉不會讓任何流程報錯,只會讓那個比例悄悄低報,所以這條是硬性契約。
3. 行尾(CRLF/LF)、UTF-8 BOM、結尾換行都照原檔還原。dets 沒改時存回去是 **byte 完全相同**,不只是欄位值相同。
4. `bbox` 一律寫成 5 位小數、clamp 到 `[0,1]`、保證 `x1<x2` / `y1<y2`。
5. 存檔後把記憶體的 bbox 回填成寫出去的值,所以「存檔 → 繼續編輯 → 再存」與「存檔 → 重開」看到的座標完全一致。

## 驗收

兩支腳本都要指向一份真實的資料夾(含 `frames/` 與 `labels/`),省略則用內建預設:

```
uv run --project D:\ws\gt_labeling python scripts/verify_roundtrip.py <gt_root>
uv run --project D:\ws\gt_labeling python scripts/verify_gui.py <gt_root>
```

預設指向 `D:\ws\detect_stream\out\gt_per_frames_0625_145125\000-020s`。那份 per-frame GT 按 `000-020s` 這樣分段,**每段自成一個 root**,要驗別段就把路徑帶上去。

**原始資料一律只讀**:兩支腳本都先把 labels(`verify_gui.py` 連 frames 一起)複製到暫存目錄,所有編輯與存檔都發生在副本上,跑完連 mtime 都不會動到原檔。

斷言一律**比相對變化量**,不寫死幀數、框數或重疊數 —— 換一份資料集就報假 FAIL 的驗收沒有價值。

- [scripts/verify_roundtrip.py](scripts/verify_roundtrip.py):存檔往返不漂移、非 dets 欄位一字未動、座標換算可逆。
- [scripts/verify_gui.py](scripts/verify_gui.py):offscreen 跑真的 GUI —— 真的開資料夾、真的用滑鼠事件改框、真的存檔再開一次比對。

## 專案結構

| 檔案                                                   | 職責                                                             |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| [main.py](main.py)                                     | 程式進入點                                                       |
| [gt_labeling/\_\_main\_\_.py](gt_labeling/__main__.py) | 命令列參數解析、QApplication 啟動                                |
| [gt_labeling/model.py](gt_labeling/model.py)           | `Det` / `FrameLabel` 資料結構、JSON 讀寫、內插、換號、undo stack |
| [gt_labeling/dataset.py](gt_labeling/dataset.py)       | 資料夾掃描、影像 LRU 快取與背景預解碼                            |
| [gt_labeling/transform.py](gt_labeling/transform.py)   | normalized ↔ widget 座標的**唯一**換算處                         |
| [gt_labeling/canvas.py](gt_labeling/canvas.py)         | 自繪影像/標註畫布,zoom / pan / hit-test / 拖曳                   |
| [gt_labeling/panel.py](gt_labeling/panel.py)           | 側邊面板:幀清單、新框預設、選取框屬性、補框設定、偵測帶          |
| [gt_labeling/window.py](gt_labeling/window.py)         | 主視窗接線:導覽、存檔、undo/redo、補框與換號流程                 |

### 設計取捨

- **自繪 QWidget 而非 QGraphicsView**:正確性關鍵是 normalized 座標不漂移。自繪只有一組 `ViewTransform`(zoom + offset),滑鼠與繪製走同一條換算;QGraphicsView 會多出 item local / scene / view 三層座標與雙向同步,正是座標漂移的溫床。代價是 zoom/pan/hit-test 自己實作約 150 行。
- **任何 normalized ↔ 螢幕換算都必須經由 `ViewTransform`**,不得在別處自行乘 `img_w * zoom` 或加 offset。
- **快照式 undo 而非命令式**:dets 每幀只有數十個小物件,複製成本可忽略;拖曳縮放這類連續操作用命令物件很容易漏記反向狀態。undo 上限 60 筆,**每幀各自一份**。
- **標註檔開啟時一次全部載入記憶體**(一幀約 1KB,逐幀資料的 600 幀也不到 1MB)。清單能立刻顯示框數與待補狀態,關掉自動存時未存編輯也能跨幀保留。
- **影像解碼結果以 LRU 快取 QPixmap**,切幀後在背景執行緒預解碼前後各 2 幀(QImage 可跨執行緒,QPixmap 只能在 GUI 執行緒建立)。3840×1920 JPEG 解碼約數十毫秒,重繪不重新解碼。
- **切幀時若影像尺寸相同就保留 zoom/pan**:逐幀比對同一區域是這工具的主要用法。

## 設定持久化

視窗大小、自動存開關、新框預設 label、補框啟用與門檻、偵測帶設定、上次開啟的資料夾,都存在 `QSettings`(organization `asys` / application `gt_labeling`)。

視窗**只還原大小、不還原位置**:記住的座標在換螢幕、改解析度或高 DPI 縮放後會落到可見範圍外,標題列抓不到就搬不動。位置交給視窗管理員,並在顯示後檢查左上角是否落在任何螢幕的可用範圍內,不在就救回來。
