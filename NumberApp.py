import io
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt, QRect, QPoint, QBuffer
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLCDNumber, QLabel

DRAWING_WIDTH = 600
DRAWING_HEIGHT = 600
padding = 10

TOP = 100
LEFT = 100
HEIGHT = DRAWING_HEIGHT + 2 * padding

clrB_x = DRAWING_WIDTH * 2 + padding * 3
clrB_y = padding
clrB_w, clrB_h = 100, 50

guessB_x = clrB_x + clrB_w + padding
guessB_y = padding
guessB_w, guessB_h = 100, 50

width = guessB_x + guessB_w + padding

lcd_x = DRAWING_WIDTH * 2 + padding * 3
lcd_y = clrB_y + clrB_h + padding
lcd_w, ldc_h = 100, 100

certl_x = DRAWING_WIDTH * 2 + padding * 3
certl_y = lcd_y + ldc_h + padding
certl_w, certl_h = 100, 200

net = tf.keras.models.load_model("models/2x16C + 64D.model")


def crop_space(img):
    img_data = np.asarray(img)
    top, bottom, left, right = -1, -1, -1, -1
    for y in range(len(img_data)):
        for x in range(len(img_data[y])):
            if img_data[y][x] != 255:
                top = y
                break
        if top != -1:
            break
    for y in range(len(img_data))[::-1]:
        for x in range(len(img_data[y])):
            if img_data[y][x] != 255:
                bottom = y
                break
        if bottom != - 1:
            break
    for x in range(len(img_data[0])):
        for y in range(top, bottom):
            if img_data[y][x] != 255:
                left = x
                break
        if left != -1:
            break
    for x in range(len(img_data[0]))[::-1]:
        for y in range(top, bottom):
            if img_data[y][x] != 255:
                right = x
        if right != -1:
            break
    return img.crop((left, top, right, bottom))


def fit_space(img, sp):
    img_data = np.asarray(img)
    new_img = np.zeros((sp, sp))
    new_img.fill(255)
    if len(img_data) == sp:
        off = sp // 2 - len(img_data[0]) // 2
        for y in range(sp):
            for x in range(sp):
                if (x < off) or (x - off >= len(img_data[0])):
                    continue
                else:
                    new_img[y][x] = img_data[y][x - off]
    else:
        off = sp // 2 - len(img_data) // 2
        for y in range(sp):
            for x in range(sp):
                if (y < off) or (y - off >= len(img_data)):
                    continue
                else:
                    new_img[y][x] = img_data[y - off][x]
    return Image.fromarray(new_img)


def format_img(img):
    img_data = np.asarray(img)
    new_img = np.zeros((len(img_data), len(img_data[0]), 1))
    for y in range(len(img_data)):
        for x in range(len(img_data[y])):
            new_img[y][x][0] = np.array([1 - (img_data[y][x] / 255)])
    return new_img


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        title = "Paint Application"

        self.setWindowTitle(title)
        self.setGeometry(TOP, LEFT, width, HEIGHT)

        self.image = QImage(DRAWING_WIDTH, DRAWING_HEIGHT, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.p_image = QImage(
            DRAWING_WIDTH, DRAWING_HEIGHT, QImage.Format_RGB32)
        self.p_image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 40
        self.brushColor = Qt.black

        self.drawingRect = QRect(padding, padding,
                                 DRAWING_WIDTH, DRAWING_HEIGHT)
        self.imageRect = QRect(DRAWING_WIDTH + padding * 2, padding,
                               DRAWING_WIDTH, DRAWING_HEIGHT)
        self.lastPoint = QPoint()

        self.clearButton = QPushButton(self)
        self.clearButton.setText("Clear")
        self.clearButton.setGeometry(clrB_x, clrB_y, clrB_w, clrB_h)
        self.clearButton.clicked.connect(self.clear)
        self.clearButton.setShortcut(Qt.Key_Z)

        self.guessButton = QPushButton(self)
        self.guessButton.setText("Guess")
        self.guessButton.setGeometry(guessB_x, guessB_y, guessB_w, guessB_h)
        self.guessButton.clicked.connect(self.change_lcd)
        self.guessButton.setShortcut(Qt.Key_X)

        self.lcd = QLCDNumber(self)
        self.lcd.setDigitCount(1)
        self.lcd.setGeometry(lcd_x, lcd_y, lcd_w, ldc_h)

        self.certl = QLabel(self)
        self.certl.setGeometry(certl_x, certl_y, certl_w, certl_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            if not self.drawingRect.contains(event.pos()):
                self.lastPoint = event.pos()
            else:
                painter = QPainter(self.image)
                painter.setPen(QPen(self.brushColor, self.brushSize,
                                    Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(QPoint(self.lastPoint.x() - padding,
                                        self.lastPoint.y() - padding),
                                 QPoint(event.x() - padding, event.y() - padding))
                self.lastPoint = event.pos()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(
            self.drawingRect, self.image, self.image.rect())
        canvas_painter.drawImage(
            self.imageRect, self.p_image, self.p_image.rect())
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        painter.drawRect(self.drawingRect)
        painter.drawRect(self.imageRect)
        self.update()

    def clear(self):
        self.image.fill(Qt.white)
        self.p_image.fill(Qt.white)
        self.update()

    def guess(self):
        bffer = QBuffer()
        bffer.open(QBuffer.ReadWrite)
        self.image.save(bffer, 'PNG')
        img = Image.open(io.BytesIO(bffer.data()))
        img = img.convert(mode='L')
        img = crop_space(img)
        img.thumbnail((20, 20))
        img = fit_space(img, 20)
        img = img.convert("L")
        self.p_image = ImageQt.ImageQt(img)
        inp = format_img(img)

        return net.predict(np.array([inp]))

    def change_lcd(self):
        res = self.guess()
        self.lcd.display(np.argmax(res))
        self.certl.setText('%\n'.join(map(lambda i: str(
            i) + ': ' + '{:f}'.format((res[0][i]) * 100)[0:5], range(10))) + '%')


def start_window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


start_window()
