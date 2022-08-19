import numpy as np
import dlib
import cv2
import pygame
import matplotlib.pyplot as plt

# FACE LANDMARK 0~67번의 점들을 부위 별로 나눈 것.
# 여기서 주로 쓰이는 건 RIGHT_EYE, LEFT_EYE이다.
# 자세한 분류는 아래 이미지 참고.
# https://user-images.githubusercontent.com/36785390/52613175-3d6ade80-2ed0-11e9-9290-ee5dc2f2d525.png
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

# FACE LANDMARK를 dlib 사용하여 학습시킨 모델.
# dlib 라이브러리에서는 얼굴 탐색을 위해 HOG특성을 활용하거나 또는 학습된 CNN(합성곱 신경망, 교재 450p)모델을 사용할 수 있다.
# 아래에서 사용할 모델은 후자이다.
# 데이터셋은 Helen dataset을 사용. - http://www.ifp.illinois.edu/~vuongle2/helen/
predictor_file = './shape_predictor_68_face_landmarks.dat'

# dlib 라이브러리를 사용해 훈련된 모델 파일 생성&불러옴.
detector = dlib.get_frontal_face_detector()         #얼굴 디텍더 초기화.
predictor = dlib.shape_predictor(predictor_file)    #얼굴의 특징점 모듈 초기화, 머신러닝으로 학습된 모델 파일 삽입.

# 눈 감고 있는 프레임을 세는 변수.
EYE_CLOSE_FRAME = 0

# EYE_CLOSE_FRAME 초기화용 프레임 수.
# 일정 시간동안 눈을 집중해서 뜨고 있으면 EYE_CLOSE_FRAME를 다시 0으로 초기화시키는 용도.
SAFE_FRAME = 0

# WAKE UP 메세지를 일정 시간동안 표시하기 위한 트리거.
WARNING_MESSAGE = False


#전방 주시를 하지 않고 있을때 증가하는 프레임 수.
FRONT_COUNT = 0

# WAKE UP 메세지 띄우는 프레임 수.
FRAME_COUNT_WAKEUP = 0

# ERE 수치를 나타내는 변수와 그 값을 저장하는 list.
ERE_num = 0
ERE_list = []

# 경고음 플레이어 변수 선언.
pygame.mixer.init()
pygame.mixer.music.load('warning.mp3')


# 경고음 출력 횟수 조절용 변수
PLAY_FLAG_1 = 0  # "WAKE UP!!"
PLAY_FLAG_2 = 0  # "LOOK FORWARD!!"

#이걸로 plt 생성.
fig = plt.figure()
#선 그래프 생성
# line1은 ERE값에 따라 실시간으로 변하는 선 그래프.
x1 = np.linspace(0, 20, 200)
y1 = np.linspace(500, 2800, 200)
line1, = plt.plot(x1, y1, 'ko-')

#line2는 ERE의 기준값인 1600을 표시하는 점선.
arr1 = [1600 for _ in range(25)]
arr2 = np.arange(0, 25)
line2, = plt.plot(arr2, arr1, 'r--')


#두 점의 거리를 반환하는 함수. d1 = [int, int]
def distDot(d1, d2):
    d = ((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)**(1/2)
    return d

#눈을 구성하는 6개의 점을 받고, 그 점의 좌표를 토대로 눈을 감았는지 떴는지를 수치화함.
#눈을 감을수록 수치가 낮게 뜸.
# 공식은 2016년 Tereza Soukupova & Jan ´ Cech에 의해 제시된 Eyes Aspect Ratio(이하 EAR) 방식을 조금 변형함.
def arrdist_EYE(arr):
    # arr[0], arr[3]이 같은 가로선에 위치하고, arr[1] arr[5], arr[2] arr[4]가 각자 같은 세로선에 위치한 점.
    return ((distDot(arr[1], arr[5]) + distDot(arr[2], arr[4])) * 2 / (distDot(arr[0], arr[3]))) * 1000


# 캠을 사용하기 위해 VideoCapture(0) 사용.
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # 현재 인식된 얼굴 수를 화면에 출력
    cv2.putText(image, "det: {}".format(len(rects)), (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for (i, rect) in enumerate(rects):

        # 얼굴 랜드마크 68개의 점을 모두 담는 행렬 points
        # 감지되는 얼굴(rects)마다 개별로 68개의 점을 모두 저장하기 위해 행렬로 선언.
        points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])

        FRONT_COUNT = 0
        show_parts = points[ALL]

        # 행렬로 받았기 때문에 np.squeeze, np.asarray를 사용해 리스트 형태로 바꿈.
        # arrdist_EYE 함수는 2차원 배열값을 받음.
        ERE_num = arrdist_EYE(np.squeeze(np.asarray(points[RIGHT_EYE]))) + arrdist_EYE(np.squeeze(np.asarray(points[LEFT_EYE])))


        #cv2.putText(image, "R_Eyes: {}".format(ERE_num), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #arrdist_EYE(np.squeeze(np.asarray(points[36:42]))) + arrdist_EYE(np.squeeze(np.asarray(points[42:48])))
        # print(points[36], points[37], points[38], points[39], points[40], points[41])
        # [[328 236]][[342 233]][[357 232]][[368 239]][[355 243]][[341 243]]

        for (i, point) in enumerate(show_parts):
            # 필요한 face landmark 좌표를 불러옴.
            x = point[0, 0]
            y = point[0, 1]

            # 68개의 얼굴 랜드마크 좌표에 따라 점을 찍음
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

            # 눈 부분 강조를 위해 던체가 아닌 눈 부분만 랜드마크 번호 출력
            if i in EYES:
                cv2.putText(image, "{}".format(i + 1), (x, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # 전방을 50프레임 이상 지켜봤다면 조는 상태가 아니라고 판단
    # 눈을 감은 시간과 SAFR_FRAME도 0으로 초기화
    if SAFE_FRAME > 35:
        SAFE_FRAME = 0
        EYE_CLOSE_FRAME = 0

    # 만약 SAFR_FRAME을 50만큼 채우지 못한 채로 눈 감은 프레임이 20을 넘어가면 운전자가 조는 것으로 판단
    # WARNING_MESSAGE를 True로 변경, 이후 경고 메세지와 경고음 출력
    if EYE_CLOSE_FRAME > 20:
        WARNING_MESSAGE = True

    # 인식된 얼굴이 하나라도 있으면
    if len(rects) != 0:
        ERE_list.append(int(ERE_num))
    else:
        ERE_list.append(500)
        # 선 그래프 데이터 업데이트.



    if len(ERE_list) < 50:
        line1.set_ydata(ERE_list)
        line1.set_xdata(np.linspace(0, 20, len(ERE_list)))
    else:
        line1.set_ydata(ERE_list[-50:-1])
        line1.set_xdata(np.linspace(0, 20, 49))

    # fig 캔버스에 다시 그리기.
    fig.canvas.draw()

    # cv2로 출력하기 위한 그래프 이미지 컨버팅.
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img는 RGB이고, cv2로 변환하기 위해 BGR로 설정.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2에서 그래프 이미지 출력.
    cv2.imshow("plot", img)


    # 1600이란 기준은 위 ERE값 특정 함수값을 통해 여러번 실험을 하면서 정한 것. 만약 과학적인 근거를 통해 수치를 조정할 수 있다면 추후에 변경 예정.
    # 실시간으로 ERE_num값 출력, 1600 이상이면 초록색, 그 아래면 빨간색 글자로 화면 왼쪽 아래에 출력.
    # 얼굴이 감지가 안되면 값이 오르지 않음.
    if ERE_num < 1700 and len(rects) >= 1:
        cv2.putText(image, "Eyes: {}".format(ERE_num), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        EYE_CLOSE_FRAME += 1
    elif ERE_num >= 1700 and len(rects) >= 1:
        cv2.putText(image, "Eyes: {}".format(ERE_num), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        SAFE_FRAME += 1
    else:
        cv2.putText(image, "Eyes: {}".format(ERE_num), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # 눈을 감고 있는 프레임, 눈을 뜨고 있는 프레임 출력.
    cv2.putText(image, "close: {}".format(EYE_CLOSE_FRAME), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(image, "safe: {}".format(SAFE_FRAME), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)


    # WAKE UP 메세지를 띄운 프레임이 15 미만이고, WARNING_MESSAGE 트리거가 True면
    # 15프레임을 다 채울 때까지 WAKE UP!! 메세지 출력, beepsound 2~3회 출력.
    if FRAME_COUNT_WAKEUP < 15 and WARNING_MESSAGE:
        FRAME_COUNT_WAKEUP += 1
        cv2.putText(image, "WAKE UP!!!!", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        PLAY_FLAG_1 += 1
        if PLAY_FLAG_1 < 2:
            pygame.mixer.music.play()

    # FRAME_COUNT_WAKEUP을 15까지 채웠으면 다시 0으로 초기화, WAKE UP 메세지 띄우지 않도록 변경.
    else:
        FRAME_COUNT_WAKEUP = 0
        PLAY_FLAG_1 = 0
        WARNING_MESSAGE = False


    # 감지된 얼굴 수가 0이면 운전자가 전방 주시를 하지 않는 걸로 간주,
    # FRONT_COUNT를 프레임 당 1씩 증가시킴.
    if len(rects) == 0:
        FRONT_COUNT += 1


    # 그렇게 15프레임 이상 전방 주시를 하지 않았다면 LOOK UP!! 메세지 출력.
    if FRONT_COUNT > 15:
        cv2.putText(image, "LOOK FORWARD!!!", (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        PLAY_FLAG_2 += 1
        if PLAY_FLAG_2 < 2:
            pygame.mixer.music.play()
    else:
        PLAY_FLAG_2 = 0

    #캠 화면 띄우기.
    cv2.imshow('Drowsy Driving Detector', image)


    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()