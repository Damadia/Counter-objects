import cv2
import numpy as np

import sobel

def getNeighborhood(x,y,img,k=3):
    h,w = img.shape
    m = k//2 * -1 #el offset ya que empezamos desde una posición anterior, aunque mantenemos valores de i,j normales para poder acceder a las entradas con normalidad
    neightborhood = np.zeros((k,k), dtype=np.uint32)

    for i in range(0,k):
        for j in range(0,k):
            #Con estás líneas podemos hacer el nearest padding
            xprime = np.clip(x+j+m,0,w-1)
            yprime = np.clip(y+i+m,0,h-1)
            neightborhood[i,j] = img[yprime,xprime]
    return neightborhood        


#Nombré mag a las MAGNITUDES DE LA IMAGEN, no es que sea un único valor, es la IMAGEN
def nonMaxSuppression(mag, theta):
    h, w = mag.shape
    out = np.zeros((h,w), dtype=np.float32)

    angle = theta * 180. / np.pi #theta es la matriz de angulos, aplicar la formula en python significa aplicar la formula a cada entrada, lo necesitamos en grados y no en radianes
    angle[angle < 0] += 180 # y el proceso es simétrico en el eje Y (podemos reflejar lo que pasa en 45° y 225° por ejemplo), no necesitamos los 360 grados

    for y in range(1, h-1):
        for x in range(1, w-1):
            q = 255
            r = 255

            #en la práctica, "quitar" lo angulos del 181° al 359° es factible porque
            #inevitablemente los vecinos del px local xy se tienen que compara con xy y no se tienen que hacer operaciones en realidad
            # solo se busca que xy sea el mayor, eso significa que, por ejemplo, los pixeles vecinos con una gradiente de 45° 
            # son los mismos px's que una gradiente de 225°

            # 0°
            if (0 <= angle[y,x] < 22.5) or (157.5 <= angle[y,x] <= 180):
                q = mag[y, x+1]
                r = mag[y, x-1]

            # 45°
            elif (22.5 <= angle[y,x] < 67.5):
                q = mag[y+1, x-1]
                r = mag[y-1, x+1]

            # 90°
            elif (67.5 <= angle[y,x] < 112.5):
                q = mag[y+1, x]
                r = mag[y-1, x]

            # 135°
            elif (112.5 <= angle[y,x] < 157.5):
                q = mag[y-1, x-1]
                r = mag[y+1, x+1]

            #q es el vecino de la "izquieda" (puede ser el de la derecha dependiendo de la dirección)
            #r es el vecion de la "derecha" (igualmente, puede ser el izquierdo)
            if mag[y,x] >= q and mag[y,x] >= r:
                out[y,x] = mag[y,x] #dado los vecinos de la dirección
                                    #sobre la que trabajamos, ver si el px local xy
                                    # es ek más grande (tiene el cambio de brillo más intenso)
                                    # entre sus dos vecinos sobre la dirección actuada.
                                    # puede ser un borde (luego el umbral define si considerarlo o no) 
            else:
                out[y,x] = 0 #suprimirlo

    return out

def threshold(img, low, high):
    strong = 255
    weak = 200

    #con esta línea suprimos a todos, lo que nos ahorra hacer la supresión después
    res = np.zeros_like(img, dtype=np.float32)

    #dos listas con las coordenadas (x,y) que cumplen la condición,
    # las x están en la lista 1, las y están en la lista 2
    # where aplica para cada elemento de la matriz 
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    #como ya todos fueron puesto en cero, no hace falta hacer esto
    # zero_i, zero_j = np.where(img < low)
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(img, weak, strong=255):
    h, w = img.shape

    for y in range(1, h-1):
        for x in range(1, w-1):
            if img[y,x] == weak:
                if np.any(img[y-1:y+2, x-1:x+2] == strong): #si alguno de los vecinos de un esquina débil es una esquina fuerte, entonces esta esquina débil se vuelve una fuerte
                    img[y,x] = strong
                else:
                    img[y,x] = 0
    return img

def cannyImg(img):
    # Suavizado
    img = cv2.GaussianBlur(img, (3,3), 1)

    # Sobel
    mag = sobel.sobelImg(img)
    theta = sobel.sobelDirs(img)

    # Non-Maximum Suppression
    nms = nonMaxSuppression(mag, theta)

    # Doble umbral
    thresh, weak, strong = threshold(nms, 1, 35)

    # Hysteresis
    edges = hysteresis(thresh, weak, strong)

    return edges

def main():
    img = cv2.imread("images/monedas1.jpg")
    cv2.imshow("Imagen original", img)

    imgG = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Imagen gris", imgG)

    edges = cannyImg(imgG)

    cv2.imshow("Canny", edges.astype(np.uint8))

    cv2.waitKey()



if __name__ == "__main__":
    main()  