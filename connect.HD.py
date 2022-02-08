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

def replaceCharDot():
    print("replaceCharDot")

#CropLogo()
#threshold_otsu()
goTemplateMatch()

'''
置換作業とは具体的にどうするのか
案１：パンニング時に、移動量を検知して、文字で隠れている画素が撮影された他のフレームの画素で置換
案２：周辺画素で補間

案２が現実的
ではどのように補間するか？
３x３カーネルで考えて、センター画素に着目。二値化画像で、センターが文字部か否かを判定。
　文字部の場合に、周辺画素で文字がない画素をカウント。
　周辺画素のうち文字でない画素の平均を作成してセンター画素を置換
　これをカーネルをスキャンしながら実施。
　？以前に置換してできた画素データは、別のセンターに対する周辺画素として利用するのか？
　　そのとき、スキャンの方向に、置換後の画像が依存する可能性がある（一定方向に引きずられる）
　　まぁ、それでもいいけど。品位は見てからかなぁ。
案３：案１と２のハイブリッド
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
