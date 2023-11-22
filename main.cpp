#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    VideoCapture capture(0);
    CascadeClassifier face_classifier;
    CascadeClassifier eye_classifier;

    // �� �� �� ���� �з��� �ε�
    face_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
    eye_classifier.load("E:/opencv/sources/data/haarcascades/haarcascade_eye.xml");

    if (!capture.isOpened()) {
        cerr << "ī�޶� �� �� �����ϴ�." << endl;
        return -1;
    }

    // ��ƼĿ �̹��� �ε�
    Mat rightEyeSticker = imread("left_eye.png");
    Mat leftEyeSticker = imread("right_eye.png");
    Mat noseSticker = imread("nose1.png");

    while (1) {
        Mat frame;
        capture >> frame;

        if (frame.empty()) {
            cerr << "�������� ��� �ֽ��ϴ�." << endl;
            continue;
        }

        Mat grayframe;
        cvtColor(frame, grayframe, COLOR_BGR2GRAY);
        equalizeHist(grayframe, grayframe);

        // �� ����
        vector<Rect> faces;
        face_classifier.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30));

        for (int i = 0; i < faces.size(); i++) {
            Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point tr(faces[i].x, faces[i].y);

            // �� �ֺ��� �簢�� �׸���
            //rectangle(frame, lb, tr, Scalar(100 * (i - 2), 255, 255 * i), 3, 4, 0);

            // �� �߽� ���
            Point faceCenter((faces[i].x + faces[i].width) / 2, (faces[i].y + faces[i].height) / 2);

            // �� ����
            Mat faceROI = grayframe(faces[i]);
            vector<Rect> eyes;
            eye_classifier.detectMultiScale(faceROI, eyes, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

            // �� ��ƼĿ �߰�
            for (size_t j = 0; j < eyes.size(); j++) {
                // ���� �߽� ��ǥ ���
                Point eyeCenter(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);

                // �� �߽��� �������� ���� ���� ������ �� ����
                if (eyeCenter.x < faceCenter.x) {  // ���� ��
                    Mat leftEyeROI = frame(Rect(faces[i].x + eyes[j].x - 50, faces[i].y + eyes[j].y - 50, eyes[j].width, eyes[j].height));
                    resize(leftEyeSticker, leftEyeSticker, Size(eyes[j].width, eyes[j].height));
                    leftEyeSticker.copyTo(leftEyeROI);
                }
                else {  // ������ ��
                    Mat rightEyeROI = frame(Rect(faces[i].x + eyes[j].x + 50, faces[i].y + eyes[j].y - 50, eyes[j].width, eyes[j].height));
                    resize(rightEyeSticker, rightEyeSticker, Size(eyes[j].width, eyes[j].height));
                    rightEyeSticker.copyTo(rightEyeROI);
                }
            }
            // �� ��ƼĿ �߰�
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