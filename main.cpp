#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    VideoCapture capture(0);
    CascadeClassifier face_classifier;
    CascadeClassifier eye_classifier;

    // 얼굴 및 눈 감지 분류기 로드
    face_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
    eye_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_eye.xml");

    if (!capture.isOpened()) {
        cerr << "카메라를 열 수 없습니다." << endl;
        return -1;
    }

    // 스티커 이미지 로드
    Mat rightEyeSticker = imread("left_eye.png");
    Mat leftEyeSticker = imread("right_eye.png");
    Mat noseSticker = imread("nose1.png");

    while (1) {
        Mat frame;
        capture >> frame;

        if (frame.empty()) {
            cerr << "프레임이 비어 있습니다." << endl;
            continue;
        }

        Mat grayframe;
        cvtColor(frame, grayframe, COLOR_BGR2GRAY);
        equalizeHist(grayframe, grayframe);

        // 얼굴 감지
        vector<Rect> faces;
        face_classifier.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30));

        for (int i = 0; i < faces.size(); i++) {
            Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point tr(faces[i].x, faces[i].y);

            // 얼굴 주변에 사각형 그리기
            //rectangle(frame, lb, tr, Scalar(100 * (i - 2), 255, 255 * i), 3, 4, 0);

            // 얼굴 중심 계산
            Point faceCenter((faces[i].x + faces[i].width) / 2, (faces[i].y + faces[i].height) / 2);

            // 눈 감지
            Mat faceROI = grayframe(faces[i]);
            vector<Rect> eyes;
            eye_classifier.detectMultiScale(faceROI, eyes, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

            // 눈 스티커 추가
            for (size_t j = 0; j < eyes.size(); j++) {
                // 눈의 중심 좌표 계산
                Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);

                // 얼굴 중심을 기준으로 왼쪽 눈과 오른쪽 눈 구분
                if (eyeCenter.x < faceCenter.x) {  // 왼쪽 눈
                    Mat leftEyeROI = frame(Rect(faces[i].x + eyes[j].x - 50, faces[i].y + eyes[j].y - 50, eyes[j].width, eyes[j].height));
                    resize(leftEyeSticker, leftEyeSticker, Size(eyes[j].width, eyes[j].height));
                    leftEyeSticker.copyTo(leftEyeROI);
                }
                else {  // 오른쪽 눈
                    Mat rightEyeROI = frame(Rect(faces[i].x + eyes[j].x + 50, faces[i].y + eyes[j].y - 50, eyes[j].width, eyes[j].height));
                    resize(rightEyeSticker, rightEyeSticker, Size(eyes[j].width, eyes[j].height));
                    rightEyeSticker.copyTo(rightEyeROI);
                }
            }
            // 코 스티커 추가
            Rect noseROI = Rect(faces[i].x + faces[i].width / 4, faces[i].y + faces[i].height / 2, faces[i].width / 2, faces[i].height / 4);
            resize(noseSticker, noseSticker, Size(noseROI.width, noseROI.height));
            noseSticker.copyTo(frame(noseROI));
        }

        imshow("webcam", frame);

        if (waitKey(30) == 27)
            break;
    }

    return 0;
}