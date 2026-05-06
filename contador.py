import cv2
import numpy as np
import queue
import canny

def neighborville(img, x, y):
    h,w = img.shape
    dt = np.dtype([('x', 'i8'), ('y', 'i8')])
    n = []

    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if i + y < 0 or i + y >= h or j + x < 0 or j + x >= w:
                continue
            n.append([i,j])
    return n

def bfs(v0, img, visited):
    # put = push; get = pop 
    list_px_connected = []
    
    q = queue.Queue()
    q.put(v0)


    while (not q.empty()):
        current_vertex = q.get()
        if visited[current_vertex[0],current_vertex[1]]:
            continue
        visited[current_vertex[0],current_vertex[1]] = True
        if img[current_vertex[0],current_vertex[1]] == 255: #Si es blanco es px es parte de un contorno
            list_px_connected.append(current_vertex)
        neigh = neighborville(img, current_vertex[0], current_vertex[1])
        for v in neigh:
            if visited[v[0],v[1]] or img[v[0],v[1]] == 0:
                continue
            q.put(v)
    return list_px_connected


def contours(img):
    h, w = img.shape
    visited = np.zeros((h,w), dtype=np.bool_) 
    visited.fill(False)

    c = []


    for y in range(h):
        for x in range(w):
            if not visited[y,x]:
                c.append(bfs([y,x], img, visited))
            else:
                continue

    return c


def amountContours(img, t=512):
    h,w = img.shape
    px_to_contours = (h*w)/t
    list_contours = contours(img)
    amount = 0

    for c in list_contours:
        if len(c) >= px_to_contours:
            amount+=1
            cv2.drawContours(img,c,-1,(0,0,255), 2)

    return amount, img


def main():
    imgOg = cv2.imread("./images/monedas1.jpg")
    imgGray = cv2.cvtColor(imgOg, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Imagen original", imgOg)
    cv2.imshow( "Imagen escala grises", imgGray)


    can = canny.cannyImg(imgGray)
    cv2.imshow( "Imagen bordes", can)
    a, imgContours  = amountContours(can)
    cv2.imshow( "Imagen contornos", imgContours)
    print("Cantidad de objetos en la imagen: ", a)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()