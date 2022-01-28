# DroidCam接続
import cv2

cap = cv2.VideoCapture()
cap.open('http://192.168.179.3:4747/video')  # OK (480,640,3)

while(True):
    ret,frame = cap.read()
    cv2.imshow("remote" , frame)
    # qキー入力でwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
cap.release()
cv2.destroyAllWindows()
