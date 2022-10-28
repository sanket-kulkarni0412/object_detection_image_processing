# import cv2
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def empty(a):
#     pass


# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("Threshold1", "Parameters", 40, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 90, 255, empty)
# cv2.createTrackbar('Area','Parameters',70,10000,empty)


# img = cv2.imread('images\ezgif.com-gif-maker.jpg')

# img_blur = cv2.GaussianBlur(img, (7, 7), 1)
# img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
# threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')
# img_canny = cv2.Canny(img_gray, threshold1, threshold2)
# kernel = np.ones((5, 5))
# img_dil = cv2.dilate(img_canny, kernel, iterations=1)

# contours, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# for contour in contours:
#     area = cv2.contourArea(contour)

#     area_min = cv2.getTrackbarPos('Area', 'Parameters')
#     if area >= area_min:
#         cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.2*peri, True)
#         print(len(approx))
#         for appro in approx:
#             x, y, w, h = cv2.boundingRect(appro)
#             cv2.rectangle(img, (x, y), (x+w, x+h), (0, 255, 0), 3)
#             cv2.putText(img, 'Points:'+str(len(appro)), (x+w+20,
#                         y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(img, 'Area:'+str(int(area)), (x+w+20, y+45),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
#             img = cv2.resize(img,(600,600))    
#     cv2.imshow('Result', img)
#     cv2.waitKey(0)

import cv2
import numpy as np
img=cv2.imread('pngtree-stacked-fresh-banana-apple-fruits-png-image_2326402.jpgs')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur_frame=cv2.GaussianBlur(img,(9,9),0)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_orange=np.array([10,50,70])
upper_orange=np.array([24,255,255])
lower_yellow=np.array([25,50,70])
upper_yellow=np.array([35, 255, 255])
lower_red=np.array([0,50,50])
upper_red=np.array([10,255,255])
mask_red=cv2.inRange(hsv,lower_red,upper_red)

mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
mask_orange=cv2.inRange(hsv,lower_orange,upper_orange)
kernel=np.ones((3,3))
mask_red=cv2.dilate(mask_red,kernel,iterations=1)
kernel=np.ones((3,3))
mask_yellow=cv2.dilate(mask_yellow,kernel,iterations=1)
kernel=np.ones((3,3))
mask_orange=cv2.dilate(mask_orange,kernel,iterations=1)
contours_apple, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_banana, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

area_list_orange=[]
contour_list_orange=[]
for contour in contours_orange:
    contour_list_orange.append(contour)
    area= cv2.contourArea(contour)
    area_list_orange.append(area)
max_orange = area_list_orange[0]
index_orange = 0
for i in range(1,len(area_list_orange)):
    if area_list_orange[i] > max_orange:
        max_orange = area_list_orange[i]
        index_orange = i
cv2.drawContours(img,contours_orange[index_orange],-1,(255,0,0),3)  
peri=cv2.arcLength(contours_orange[index_orange],True)
approx_orange=cv2.approxPolyDP(contours_orange[index_orange],0.001*peri,True   )
x,y,w,h=cv2.boundingRect(approx_orange)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.putText(img,'Orange',(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)
cv2.putText(img,'Area:'+str(int(area)),(x+w+20,y+45),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)

area_list_apple=[]
contour_list_apple=[]
for contour in contours_apple:
    contour_list_apple.append(contour)
    area= cv2.contourArea(contour)
    area_list_apple.append(area)
max = area_list_apple[0]
index = 0
for i in range(1,len(area_list_apple)):
    if area_list_apple[i] > max:
        max = area_list_apple[i]
        index = i
cv2.drawContours(img,contours_apple[index],-1,(255,0,0),3)  
peri=cv2.arcLength(contours_apple[index],True)
approx=cv2.approxPolyDP(contours_apple[index],0.001*peri,True   )
x,y,w,h=cv2.boundingRect(approx)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.putText(img,'Apple',(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)
cv2.putText(img,'Area:'+str(int(area)),(x+w+20,y+45),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)

sort_list=[]
area_list_banana=[]
contour_list_banana=[]
for contour in contours_banana:
    contour_list_banana.append(contour)
    area_banana= cv2.contourArea(contour)
    area_list_banana.append(area_banana)
    sort_list.append(area_banana)
sort_list.sort()
print(sort_list)
value_=sort_list[-1]
index_banana=area_list_banana.index(value_)
print(index_banana)
# max_banana = area_list_banana[0]
# index_banana = 0
# for i in range(1,len(area_list_banana)):
#     if area_list_banana[i] > max:
#         max = area_list_banana[i]
#         index = i
cv2.drawContours(img,contours_banana[index_banana],-1,(255,0,0),3)  
peri=cv2.arcLength(contours_banana[index_banana],True)
approx=cv2.approxPolyDP(contours_banana[index_banana],0.001*peri,True   )
x,y,w,h=cv2.boundingRect(approx)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.putText(img,'Banana',(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)
cv2.putText(img,'Area:'+str(int(area)),(x+w+20,y+45),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)



#cv2.drawContours(img,contours,-1,(0,255,0),2)
img=cv2.resize(img,(1000,1000))
mask_red=cv2.resize(mask_red,(600,600))
cv2.imshow('img',img)
cv2.imshow('mask',mask_red)
cv2.waitKey(0)