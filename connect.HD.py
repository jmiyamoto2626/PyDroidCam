# DroidCam接続
from logging.handlers import RotatingFileHandler
from tracemalloc import start
import cv2
import numpy as np


import time

# 前処理：矩形指定でロゴ部を切り出す
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


# 前処理：ロゴ部を２値化（背景が黒、文字が白）
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


# 本処理：テンプレートマッチでロゴのある場所を特定して切り出し
# ロゴの場所が接続単位でしか移動しない場合は、接続初期だけの実行でよい
def goTemplateMatch(ref,cmp):
    debug = 0
    if debug:
        print("goTemplateMatch")
    #ref = cv2.imread("logo.thresh.bmp")
    #cmp = cv2.imread("finish_frame.room.jpg")
    ref_height , ref_width = ref.shape[:2]

    res = cv2.matchTemplate(cmp,ref,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    x,y = max_loc
    match_img = cmp[y:y+ref_height , x:x+ref_width]
    if debug:
        cv2.imshow("match",match_img)
        cv2.imshow("ref",ref)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #cv2.imwrite("match_img.bmp",match_img)
    return max_loc,match_img

# 前処理：ロゴ部の２値化画像を白黒反転（背景：白、文字：黒）
def mono2whiteback(ref):
    debug = 0
    #ref = cv2.imread("logo.thresh.bmp")
    dst = cv2.bitwise_not(ref)
    if debug:
        cv2.imshow("before",ref)
        cv2.imshow("after",dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #cv2.imwrite("logo.thresh.inv.bmp",dst)
    return dst

# 本処理：文字部を置換して文字を消す（誤魔化す）
def replaceCharDot(mono,src):
    #print("replaceCharDot")
    debug = 0
    speed_test = 1
    if speed_test:
        start_time =time.time()
    size_mono = 3,3,3
    size_color = 3,3,3
    kernel3x3mono = np.zeros(size_mono,dtype=np.uint8)
    kernel3x3color = np.zeros(size_color,dtype=np.uint8)
    
    if debug:
        print("kernel3x3mono.shape:",kernel3x3mono.shape)
    #mono = cv2.imread("logo.thresh.inv.bmp")
    #src = cv2.imread("match_img.bmp")
    #mono_org = mono.copy()
    #src_org = src.copy()
    mono_mod = mono.copy()
    width = src.shape[1]
    height = src.shape[0]
    if debug :
        print("width , height =",width , height)
    for n in range (height -2):
        if debug :
            print("---n=",n,"---")
        for m in range(width -2):
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
                    cv2.imshow("debug mono",kMonoResize)
                    cv2.imshow("debug color",kColorResize)

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
                if count_valid != 0:
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
                mono_mod[n+1,m+1,0] = 255
                mono_mod[n+1,m+1,1] = 255
                mono_mod[n+1,m+1,2] = 255
            # 水平スキャンの終端
        # 垂直スキャンの終端
    # 置換作業の終端
    if debug :
        cv2.imshow("fin_mono",mono_mod)
        cv2.imshow("fin_color",src)
        #cv2.imshow("org_mono",mono_org)
        #cv2.imshow("org_color",src_org)
        #cv2.waitKey(1)
    #cv2.imwrite("mono.replaced.bmp",mono)
    #cv2.imwrite("color.replaced.bmp",src)
    if speed_test:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return src
# 関数の終端

def ReplaceLogo(src):
    debug = 1
    if debug:
        print("ReplaceLogo")



# ここからメイン処理

# ロゴ部のファイル読み出し（２値化されたビットマップ）
ref = cv2.imread("logo.thresh.bmp")
# ロゴ部の白黒反転（背景：白、文字：黒）
ref_inv = mono2whiteback(ref)

# キャプチャの初期設定
cap = cv2.VideoCapture()
cap.open('http://192.168.179.3:4747/video/force/1280x720') # OK(1280 , 640 , 3)

# 最初に１フレーム取り込み
ret0,frame0 = cap.read()
# ロゴの位置を取得
max_loc,match_img0 = goTemplateMatch(ref,frame0)
x,y = max_loc
ref_height , ref_width = ref.shape[:2]

print("max_loc:",max_loc)
print("ref_height,ref_width:",ref_height,ref_width)
#cv2.imshow("match_img0",match_img0)

count = 0
while(True):
    count = count + 1
    ret,frame = cap.read()
    # ロゴ部を切り出し
    #if count == 100:
    #    cv2.imwrite("100.before.bmp",frame)
    match_img = frame[y:y+ref_height , x:x+ref_width]
    replaced_logo = replaceCharDot(ref_inv,match_img)
    #cv2.imshow("replaced_logo",replaced_logo)
    #cv2.imshow("ref_inv",ref_inv)

    # 文字部を置換したもので原画に上書きする
    frame[y:y+ref_height , x:x+ref_width] = replaced_logo
    cv2.imshow("remote" , frame)
    #if count == 100:
    #    cv2.imwrite("100.after.bmp",frame)
    # qキー入力でwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite("finish_frame.jpg",frame)
        #cv2.imwrite("finish_frame.bmp",frame)
        break
    
# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()

