
import numpy as np
import cv2
import os
import random

file = ['088843.jpg']
output = 'fj-wu.png'

img = cv2.imread('088843.jpg')
fog = cv2.imread('fog/fog.png')

h,w,_ = img.shape

fog= cv2.resize(fog,dsize=(w,h))

for i in range(40):
    fog_num = round(random.uniform(0.03, 0.6), 3)
    result = cv2.addWeighted(img, fog_num , fog,1,0)


    cv2.imwrite('result/fj-{}.png'.format(fog_num), result)

# cv2.imshow('result',result)
# cv2.waitKey(0)


print(h,w);quit()




