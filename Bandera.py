import cv2
import numpy as np
import matplotlib.pyplot as plt
from hough import hough
from orientation_estimate import *
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


class Bandera:  # definimos la clase bandera

    def __init__(self, image):  # el constructor
        self.image = image  # se carga la imagen y se almacena en imagen con self
        self.image = np.array(image, dtype=np.float64) / 255
        #cv2.imshow("image",self.image)
        #cv2.waitKey(0)
    def colores(self,color): #método que imprime los colores de la bandera en consola
        self.color = color
        # transforma la imagen en un array 2D np.
        self.rows, self.cols, self.ch = self.image.shape
        assert self.ch == 3
        self.image_array = np.reshape(self.image, (self.rows * self.cols, self.ch))
        self.image_array_sample = shuffle(self.image_array, random_state=0)[:10000]
        self.model = KMeans(n_clusters=color, random_state=0).fit(self.image_array_sample)
        self.labels = self.model.predict(self.image_array)
        self.centers = self.model.cluster_centers_
        self.d = self.centers.shape[1]
        self.image_clusters = np.zeros((self.rows, self.cols, self.d))
        self.label_idx = 0
        for i in range(self.rows):
            for j in range(self.cols):
                self.image_clusters[i][j] = self.centers[self.labels[self.label_idx]]
                self.label_idx += 1
        return self.image_clusters

    def porcentaje(self,image):  # método que imprime los colores de la bandera en consola
        self.image =image
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.hist_hsv = cv2.calcHist([self.image_hsv], [0], None, [180], [0, 180])  # imagen, canal, máscara, tamaño, rango
        plt.plot(self.hist_hsv, color='red')
        plt.xlim([0, 180])
        plt.show()
    def orientacion(self):
        self.high_thresh = 100  # Nivel de líneas (a mayor menos líneas)
        self.bw_edges = cv2.Canny(self.image, self.high_thresh * 0.3, self.high_thresh, L2gradient=True)

        self.hough = hough(self.bw_edges)
        self.accumulator = self.hough.standard_HT()

        self.acc_thresh = 50
        self.N_peaks = 11
        self.nhood = [25, 9]
        self.peaks = self.hough.find_peaks(self.accumulator, self.nhood, self.acc_thresh, self.N_peaks)

        [self._, self.cols] = self.image.shape[:2]
        self.image_draw = np.copy(self.image)
        for i in range(len(self.peaks)):
            self.rho = self.peaks[i][0]
            self.theta_ = self.hough.theta[self.peaks[i][1]]

            self.theta_pi = np.pi * self.theta_ / 180
            self.theta_ = self.theta_ - 180
            self.a = np.cos(self.theta_pi)
            self.b = np.sin(self.theta_pi)
            self.x0 = self.a * self.rho + self.hough.center_x
            self.y0 = self.b * self.rho + self.hough.center_y
            self.c = -self.rho
            self.x1 = int(round(self.x0 + self.cols * (-self.b)))
            self.y1 = int(round(self.y0 + self.cols * self.a))
            self.x2 = int(round(self.x0 - self.cols * (-self.b)))
            self.y2 = int(round(self.y0 - self.cols * self.a))

            if np.abs(self.theta_) < 80:
                self.image_draw = cv2.line(self.image_draw, (self.x1, self.y1), (self.x2, self.y2), [0, 255, 255], thickness=2)
            elif np.abs(self.theta_) > 100:
                self.image_draw = cv2.line(self.image_draw, (self.x1, self.y1), (self.x2, self.y2), [255, 0, 255], thickness=2)
            else:
                if self.theta_ > 0:
                    self.image_draw = cv2.line(self.image_draw, (self.x1, self.y1), (self.x2, self.y2), [0, 255, 0], thickness=2)
                else:
                    self.image_draw = cv2.line(self.image_draw, (self.x1, self.y1), (self.x2, self.y2), [0, 0, 255], thickness=2)
        cv2.imshow("frame", self.bw_edges)
        cv2.imshow("lines", self.image_draw)
        cv2.waitKey(0)
        return self.theta_



