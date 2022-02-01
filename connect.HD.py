# DroidCam接続
from logging.handlers import RotatingFileHandler
import cv2

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

#CropLogo()

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

threshold_otsu()

'''
テンプレートマッチングする
マッチした座標を記憶
毎フレーム、指定の位置で置換作業をする

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
