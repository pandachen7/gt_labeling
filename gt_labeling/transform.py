"""normalized <-> widget 座標的唯一換算處。

不變量(改動本檔以外的地方時請維持):

* JSON 的 ``bbox`` 是 normalized ``[0,1]`` float,**那是唯一真值**;畫面像素只是投影。
* 任何 normalized <-> 螢幕的換算都必須經由 :class:`ViewTransform`,不得在別處自行
  乘 ``img_w * zoom`` 或加 offset —— 這是這類工具座標漂移的主要來源。
* 只有兩個狀態:``zoom``(影像 px -> widget px 的倍率)與 ``off_x/off_y``(影像左上角
  落在 widget 的位置)。縮放時 offset 由「游標下的 normalized 點保持不動」重新推導,
  而非累加修正,因此連續縮放不累積誤差。
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, QSize, QSizeF

MIN_ZOOM = 0.02
MAX_ZOOM = 40.0


@dataclass
class ViewTransform:
    img_w: int = 1
    img_h: int = 1
    zoom: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0

    # ------------------------------------------------------------------ 影像尺寸

    def set_image_size(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"影像尺寸必須為正:{w}x{h}")
        self.img_w = int(w)
        self.img_h = int(h)

    @property
    def span_x(self) -> float:
        """normalized 1.0 對應多少 widget 像素(水平)。"""
        return self.img_w * self.zoom

    @property
    def span_y(self) -> float:
        return self.img_h * self.zoom

    # ------------------------------------------------------- normalized -> widget

    def n2v(self, nx: float, ny: float) -> QPointF:
        return QPointF(nx * self.span_x + self.off_x, ny * self.span_y + self.off_y)

    def n2v_rect(self, bbox) -> QRectF:
        """bbox 允許 x1>x2(拖曳中交叉),回傳已 normalized 的 QRectF。"""
        return QRectF(self.n2v(bbox[0], bbox[1]), self.n2v(bbox[2], bbox[3])).normalized()

    # ------------------------------------------------------- widget -> normalized

    def v2n(self, vx: float, vy: float) -> QPointF:
        """回傳值可能超出 [0,1](游標在影像外),由呼叫端決定是否 clamp。"""
        return QPointF((vx - self.off_x) / self.span_x, (vy - self.off_y) / self.span_y)

    def v2n_point(self, p: QPointF) -> QPointF:
        return self.v2n(p.x(), p.y())

    def v2n_rect(self, rect: QRectF) -> list[float]:
        r = rect.normalized()
        a = self.v2n(r.left(), r.top())
        b = self.v2n(r.right(), r.bottom())
        return [a.x(), a.y(), b.x(), b.y()]

    def px_to_n(self, px: float) -> QSizeF:
        """widget 像素長度換成 normalized 長度(x/y 因影像長寬比而不同)。"""
        return QSizeF(px / self.span_x, px / self.span_y)

    # ------------------------------------------------------------------- 影像範圍

    def image_rect(self) -> QRectF:
        return QRectF(self.off_x, self.off_y, self.span_x, self.span_y)

    # --------------------------------------------------------------- 縮放與平移

    def fit_zoom(self, view: QSize) -> float:
        if view.width() <= 0 or view.height() <= 0:
            return self.zoom
        return min(view.width() / self.img_w, view.height() / self.img_h)

    def fit(self, view: QSize) -> None:
        if view.width() <= 0 or view.height() <= 0:
            return
        self.zoom = max(MIN_ZOOM, min(self.fit_zoom(view), MAX_ZOOM))
        self.center(view)

    def center(self, view: QSize) -> None:
        self.off_x = (view.width() - self.span_x) / 2.0
        self.off_y = (view.height() - self.span_y) / 2.0

    def zoom_by(self, factor: float, anchor: QPointF, min_zoom: float = MIN_ZOOM) -> bool:
        """以 ``anchor``(widget 座標)為錨點縮放,回傳 zoom 是否真的改變。"""
        lo = max(MIN_ZOOM, min_zoom)
        target = min(max(self.zoom * factor, lo), MAX_ZOOM)
        if target == self.zoom:
            return False
        anchor_n = self.v2n_point(anchor)
        self.zoom = target
        self.off_x = anchor.x() - anchor_n.x() * self.span_x
        self.off_y = anchor.y() - anchor_n.y() * self.span_y
        return True

    def pan_by(self, dx: float, dy: float) -> None:
        self.off_x += dx
        self.off_y += dy

    def clamp_offset(self, view: QSize) -> None:
        """影像比視窗小則置中,比視窗大則不許拖出視窗外。"""
        if view.width() > 0:
            if self.span_x <= view.width():
                self.off_x = (view.width() - self.span_x) / 2.0
            else:
                self.off_x = min(max(self.off_x, view.width() - self.span_x), 0.0)
        if view.height() > 0:
            if self.span_y <= view.height():
                self.off_y = (view.height() - self.span_y) / 2.0
            else:
                self.off_y = min(max(self.off_y, view.height() - self.span_y), 0.0)
