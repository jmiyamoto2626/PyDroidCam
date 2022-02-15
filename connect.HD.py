# DroidCam接続
from logging.handlers import RotatingFileHandler
from tracemalloc import start
import cv2
import numpy as np


import time

def CropLogo():
    src = cv2.imread("./finish_frame.black.bmp")

    rect = cv2.selectROI("select ROI",src)
    x,y,width,height = rect
    print(rect)
    dst = src[y:y+height , x:x+width]
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("logo.bmp",dst)


def threshold_otsu():
    src = cv2.imread("logo.bmp")
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh,dst = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    print(thresh)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("logo.thresh.bmp",dst)


'''
テンプレートマッチングする
マッチした座標を記憶
毎フレーム、指定の位置で置換作業をする

'''

def goTemplateMatch():
    ref = cv2.imread("logo.thresh.bmp")
    #cmp = cv2.imread("finish_frame.black.jpg")
    #cmp = cv2.imread("finish_frame.red.jpg")
    cmp = cv2.imread("finish_frame.room.jpg")
    #print(ref.shape)
    #print(cmp.shape)
    ref_height , ref_width = ref.shape[:2]
    #print(ref_height , "," , ref_width)

    res = cv2.matchTemplate(cmp,ref,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #print(max_loc)
    x,y = max_loc
    #print(y,",",x)
    match_img = cmp[y:y+ref_height , x:x+ref_width]
    #print(match_img.shape[:2])
    cv2.imshow("match",match_img)
    cv2.imshow("ref",ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("match_img.bmp",match_img)

def mono2whiteback():
    ref = cv2.imread("logo.thresh.bmp")
    dst = cv2.bitwise_not(ref)
    cv2.imshow("before",ref)
    cv2.imshow("after",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("logo.thresh.inv.bmp",dst)

def replaceCharDot():
    print("replaceCharDot")
    debug = 0
    size_mono = 3,3,3
    size_color = 3,3,3
    kernel3x3mono = np.zeros(size_mono,dtype=np.uint8)
    kernel3x3color = np.zeros(size_color,dtype=np.uint8)
    
    if debug:
        print("kernel3x3mono.shape:",kernel3x3mono.shape)
    mono = cv2.imread("logo.thresh.inv.bmp")
    src = cv2.imread("match_img.bmp")
    mono_org = mono.copy()
    src_org = src.copy()
    width = src.shape[1]
    height = src.shape[0]
    if debug :
        print("width , height =",width , height)
    for n in range (height -2):
    #for n in range (3):
        if debug :
            print("---n=",n,"---")
        for m in range(width -2):
        #for m in range(3):
            if debug :
                print("-----n:m=",n,":",m,"-----")
            kernel3x3mono[0,0] = mono[n,m]
            kernel3x3mono[0,1] = mono[n,m+1]
            kernel3x3mono[0,2] = mono[n,m+2]
            kernel3x3mono[1,0] = mono[n+1,m]
            kernel3x3mono[1,1] = mono[n+1,m+1]
            kernel3x3mono[1,2] = mono[n+1,m+2]
            kernel3x3mono[2,0] = mono[n+2,m]
            kernel3x3mono[2,1] = mono[n+2,m+1]
            kernel3x3mono[2,2] = mono[n+2,m+2]
            
            #exit
            if 0 :
                cv2.imshow("debug mono",kernel3x3mono)
                cv2.imshow("debug color",kernel3x3color)
                #cv2.waitKey(0)
            if kernel3x3mono[1,1,0] == 0 :
                # センターが黒（文字）画素だったら
                kernel3x3color[0,0] = src[n,m]
                kernel3x3color[0,1] = src[n,m+1]
                kernel3x3color[0,2] = src[n,m+2]
                kernel3x3color[1,0] = src[n+1,m]
                kernel3x3color[1,1] = src[n+1,m+1]
                kernel3x3color[1,2] = src[n+1,m+2]
                kernel3x3color[2,0] = src[n+2,m]
                kernel3x3color[2,1] = src[n+2,m+1]
                kernel3x3color[2,2] = src[n+2,m+2]
                kernel3x3color = cv2.bitwise_and(kernel3x3mono,kernel3x3color)

                if debug : 
                    print(kernel3x3mono)
                    print("---")
                    print(kernel3x3color)
                if debug :
                    size_resize = 300,300,3
                    kMonoResize = np.zeros(size_resize,dtype=np.uint8)
                    kColorResize = np.zeros(size_resize,dtype=np.uint8)
                    kMonoResize[  0:100,  0:100] = kernel3x3mono[0,0]
                    kMonoResize[  0:100,100:200] = kernel3x3mono[0,1]
                    kMonoResize[  0:100,200:300] = kernel3x3mono[0,2]
                    kMonoResize[100:200,  0:100] = kernel3x3mono[1,0]
                    kMonoResize[100:200,100:200] = kernel3x3mono[1,1]
                    kMonoResize[100:200,200:300] = kernel3x3mono[1,2]
                    kMonoResize[200:300,  0:100] = kernel3x3mono[2,0]
                    kMonoResize[200:300,100:200] = kernel3x3mono[2,1]
                    kMonoResize[200:300,200:300] = kernel3x3mono[2,2]
                    
                    kColorResize[  0:100,  0:100] = kernel3x3color[0,0]
                    kColorResize[  0:100,100:200] = kernel3x3color[0,1]
                    kColorResize[  0:100,200:300] = kernel3x3color[0,2]
                    kColorResize[100:200,  0:100] = kernel3x3color[1,0]
                    kColorResize[100:200,100:200] = kernel3x3color[1,1]
                    kColorResize[100:200,200:300] = kernel3x3color[1,2]
                    kColorResize[200:300,  0:100] = kernel3x3color[2,0]
                    kColorResize[200:300,100:200] = kernel3x3color[2,1]
                    kColorResize[200:300,200:300] = kernel3x3color[2,2]
                    #kMonoResize = cv2.resize(kernel3x3mono,dsize=None,fx=100,fy=100)
                    #kColorResize = cv2.resize(kernel3x3color,dsize=None,fx=100,fy=100)
                    cv2.imshow("debug mono",kMonoResize)
                    cv2.imshow("debug color",kColorResize)
                    #cv2.waitKey(100)

                # 白（有効画素）をカウント
                count_valid = 0
                for i in range(3):
                    for j in range(3):
                        if kernel3x3mono[i,j,0] == 255:
                            count_valid = count_valid + 1
                if debug :
                    print("count_valid:",count_valid)
                # 有効画素のレベルの積算値を算出。rgbごとに。
                color_sum_b = 0
                color_sum_g = 0
                color_sum_r = 0
                for i in range(3):
                    for j in range(3):
                        color_sum_b = color_sum_b + kernel3x3color[i,j,0]
                        color_sum_g = color_sum_g + kernel3x3color[i,j,1]
                        color_sum_r = color_sum_r + kernel3x3color[i,j,2]
                # 有効画素の平均レベルを算出
                color_ave_b = color_sum_b / count_valid
                color_ave_g = color_sum_g / count_valid
                color_ave_r = color_sum_r / count_valid
                if debug :
                    print("color_ave(b,g,r):",color_ave_b,color_ave_g,color_ave_r)
                # カラー画像の中央を平均値で置換
                src[n+1,m+1,0] = color_ave_b
                src[n+1,m+1,1] = color_ave_g
                src[n+1,m+1,2] = color_ave_r
                # ２値化画像の中央を有効画素扱いにする（置換が終わったから）
                mono[n+1,m+1,0] = 255
                mono[n+1,m+1,1] = 255
                mono[n+1,m+1,2] = 255
            # 水平スキャンの終端
        # 垂直スキャンの終端
    # 置換作業の終端
    if debug :
        cv2.imshow("fin_mono",mono)
        cv2.imshow("fin_color",src)
        cv2.imshow("org_mono",mono_org)
        cv2.imshow("org_color",src_org)
        cv2.waitKey(1)
    cv2.imwrite("mono.replaced.bmp",mono)
    cv2.imwrite("color.replaced.bmp",src)

                
                
                        
                
                    




        

start_time = time.time()
#CropLogo()
#threshold_otsu()
#mono2whiteback()
#goTemplateMatch()
replaceCharDot()
end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

'''
２値化画像を文字を黒、背景を白にする
文字部を拡張処理する（でこぼこを無くする）
３x３カーネルで２値化画像をスキャンする
センターに文字（黒）がある場合、
　　カラー画像の同じ位置の３x３に対して２値化画像でANDをとることで有効画素のみ取得
　　有効画素を有効画素数で割って平均を算出
　　カラー画像のカーネル中央を平均値で置換
　　２値化画像のカーネル中央を背景（白）で置換
　　次に進む
右端まで行ったら、左端に戻って位置画素下に移動
これを繰り返して切り出し後の全画素をスキャンしながら置換を完了する

置換が終わったら、元画像に対して貼り付けする
'''

'''
cap = cv2.VideoCapture()
#cap.open('http://192.168.179.3:4747/video')  # OK (480,640,3)
cap.open('http://192.168.179.3:4747/video/force/1280x720') # OK(1280 , 640 , 3)

while(True):
    ret,frame = cap.read()
    cv2.imshow("remote" , frame)
    # qキー入力でwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite("finish_frame.jpg",frame)
        cv2.imwrite("finish_frame.bmp",frame)
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()

'''
