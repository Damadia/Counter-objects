import cv2
import numpy as np



# Checar si el kernel se sale de la imagen para hacer el Padding simétrico,
# Como se usa el padding simétrico, eso hará que las mascaras en Ix y Iy sean 0
# dicho de otra forma, como las diferencias en x e y se hacen 0, no hay gradiente, 
# y por lo tanto, los límites de la imagen no son considerados
def checkOutside(x,y,size):
    flags = np.zeros(4, dtype=np.int16)

    # size[0] = h, size[1] = w 
    if (x < 0): # Desbordamiento izq 
        flags[0] = 1
    if (x >= size[1]): # Desbordamiento der 
        flags[1] = 1
    if (y < 0): # Desbordamiento arr 
        flags[2] = 1
    if (y >= size[0]): # Desbordamiento abj 
        flags[3] = 1

    return flags

def gradientX(n, k=3):
    # Sobel busca las derivas parciales de primer orden de x e y, de la misma manera que prewitt
    # lo que lo hace diferente es que ahora el kernel no es solo de 3x3, es kxk y se le asignan pesos a la diferencias finitas.
    # estos pesos son más altos conforme se acercal a pixel local xy para tener mayor consideración a la información en dicho px
    # La razón de ser del Kernel y su apariencia que tiene en x e y son las mismas que prewitt pero ahora se atribuyen pesos 
    # cuando se usar la información de tres px's, se necesitan 3 filas/columnas, veamos el ejemplo con la fila central para la gradiente
    # en X (también se debe hacer lo mismo con la 1ra y 3ra fila) e.g:
    # dI/dx1 = 2*I(x2,y) - 2*I(x1,y) <- la primera
    # dI/dx0 = 2*I(x1, y) - 2*I(x0,y) <- la segunda
    # Sumando
    # dI/dx1 + dI/dx0 = 2*I(x2,y) - 2*I(x1,y) + 2*I(x1, y) - 2*I(x0,y) = 2*I(x2,y) - 2*I(x0,y)
    # Y vemos como se cancela el centro, pero también hay pesos de 2 porque es el px central y el más importante
    # la misma lógica se aplica para las otras filas y para las columnas con el kernel para la Gy

    #Construir el Kernel
    kernel = np.zeros((k,k),np.int32)
    m = k//2
    r=1
    r2 = 1
    flag = False
    for i in range(0, k):
        for j in range(0, k):
            if j < m:
                if i <= m:
                    kernel[i,j] = -1*((i+1)*(1+j))
                else:
                    kernel[i,j] = kernel[m-r2,j]
                    flag = True
            elif j == m:
                kernel[i,j] = 0
            else:
                kernel[i,j] = -1*kernel[i, m-r]
                r+=1
        r = 1
        if flag:
            r2 += 1
            flag = False
    

    # Obtener el valor de la componente X de la gradiente Delta con las 
    # diferencias finitas con ayuda del kernel de convulsión ya construido
    Gx = 0
    for i in range(0, k):
        for j in range(0, k):
            Gx += kernel[i,j]*n[i,j]


    return Gx / np.sum(np.abs(kernel)) #normalizarlo 

            
def gradientY(n,k=3):
    kernel = np.zeros((k,k),np.int32)
    m = k//2
     # Llenar la matriz en la parte superior izquieda
    for i in range(m):
        for j in range(m+1):
            kernel[i, j] = -((1 + i) * (j + 1))

    r=1
    #reflejo en horizontal
    for i in range(0,m):
        for j in range(m+1,k):
            kernel[i,j] = kernel[i, m-r]
            r+=1
        r = 1
    r=1
    #reflejo vertical 
    for i in range(m + 1, k):
        for j in range(k):
            kernel[i, j] = -kernel[m-r, j]
        r+=1

    
    #print(kernel)
    # Obtener el valor de la componente X de la gradiente Delta con las 
    # diferencias finitas con ayuda del kernel de convulsión ya construido
    Gy = 0
    for i in range(0, k):
        for j in range(0, k):
            Gy += kernel[i,j]*n[i,j]

    #Se cancela en la normalización por lo negativos ARREGLA ESO    
    return Gy / np.sum(np.abs(kernel)) #normalizarlo 

def gradient(n, k=3):
    #Sé que puedo hacer una única llamada y calcular las dos gradientes en un solo loop
    #pero para mayor claridad visual lo deje en dos llamadas
    Gx = gradientX(n, k=3)
    Gy = gradientY(n ,k=3)

    #y aproximamos la magnitud con el valor absoluto
    GMag = np.abs(Gx) + np.abs(Gy)
    GDir = np.arctan2(Gy, Gx)
    return GMag, GDir


def getNeighborhood(x,y,img,k=3):
    h,w = img.shape
    m = k//2 * -1 #el offset ya que empezamos desde una posición anterior, aunque mantenemos valores de i,j normales para poder acceder a las entradas con normalidad
    neightborhood = np.zeros((k,k), dtype=np.int32)

    for i in range(0,k):
        for j in range(0,k):
            #Con estás líneas podemos hacer el nearest padding
            xprime = np.clip(x+j+m,0,w-1)
            yprime = np.clip(y+i+m,0,h-1)
            neightborhood[i,j] = img[yprime,xprime]
    return neightborhood        
        
def sobelImg(img,k=3):
    p = img.copy()
    h,w = img.shape
    n = np.zeros((k,k), dtype=np.float32)

    for y in range(0, h):
        for x in range(0, w):
            n = getNeighborhood(x,y,img)
            p[y,x] = gradient(n,k=3)[0] #aquí lo que se regresa es la magnitud, en sobel si podemos definir una k

    return p


def sobelDirs(img, k=3):
    h, w = img.shape
    p = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h):
        for x in range(0, w):
            n = getNeighborhood(x, y, img)
            p[y, x] = gradient(n, k=3)[1]
    
    return p

def edges(pImg,t=128):
    h,w = pImg.shape
    edgeImg = pImg.copy()
    for y in range(0,h-1):
        for x in range(0,w-1):
            if pImg[y,x] < t:
                edgeImg[y,x] = 0
            else:
                edgeImg[y,x] = 255

    return edgeImg




def main():
    img = cv2.imread("images/metal_archie_comics.jpg")
    cv2.imshow("Imagen original",img)    
    imgG = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Imagen Gris", imgG)
    imgP = sobelImg(imgG)
    cv2.imshow("Imagen gradiente", imgP)
    edgeImg = edges(imgP,50)
    cv2.imshow("Imagen bordes", edgeImg)

    cv2.waitKey()



if __name__ == "__main__":
    main()