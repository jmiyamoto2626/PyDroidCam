# DroidCam接続
import cv2

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
