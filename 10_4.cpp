
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;



void detectFaces(const Mat& frame, CascadeClassifier& face_cascade_face, vector<Rect>& faces) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    face_cascade_face.detectMultiScale(gray, faces, 1.1, 5);// , 0, Size(30, 30));

    for (const auto& face : faces) {
        rectangle(frame, face, Scalar(0, 0, 255), 2);
    }
}

void detectEyes(const Mat& frame, const vector<Rect>& faces, CascadeClassifier& face_cascade_eye) {
    for (const auto& face : faces) {
        Mat faceS = frame(face);

        vector<Rect> eyes;
        face_cascade_eye.detectMultiScale(faceS, eyes, 1.1, 35, 0, cv::Size(10, 10), Size(50,50));

        for (const auto& eye : eyes) {
            cv::Point center(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
            int radius = cvRound((eye.width + eye.height) * 0.25);
            circle(frame, center, radius, Scalar(255, 0, 0), 2);
        }
    }
}

void detectSmiles(const Mat& frame, const vector<Rect>& faces, CascadeClassifier& face_cascade_smile) {
    for (const auto& face : faces) {
        Mat faceS = frame(face);

        vector<Rect> smiles;
        face_cascade_smile.detectMultiScale(faceS, smiles, 1.5, 28, 0, Size(50, 50), Size(130,130));

        for (const auto& smile : smiles) {
            rectangle(frame, Point(face.x + smile.x, face.y + smile.y), Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height), Scalar(0, 255, 0), 2);
        }
    }
}

int main() {

    CascadeClassifier face_cascade_eye;
    if (!face_cascade_eye.load("C:/Users/veras/Desktop/9.04/data/haarcascades/haarcascade_eye.xml")) {
        cout << "File Error" << endl;
        return -1;
    }

    CascadeClassifier face_cascade_smile;
    if (!face_cascade_smile.load("C:/Users/veras/Desktop/9.04/data/haarcascades/haarcascade_smile.xml")) {
        cout << "File Error" << endl;
        return -1;
    }


    CascadeClassifier face_cascade_face;
    if (!face_cascade_face.load("C:/Users/veras/Desktop/9.04/data/haarcascades/haarcascade_frontalface_default.xml")) {
        cout << "File Error" << endl;
        return -1;
    }

    VideoCapture cap("C:/Users/veras/Desktop/ZUA.mp4");
    if (!cap.isOpened()) {
        std::cerr << "error" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    vector<Mat> processedFrames;

    auto start = chrono::steady_clock::now();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {

            break;
        }

        Mat resizedFrame;
        resize(frame, resizedFrame, Size(), 0.75, 0.75, INTER_LANCZOS4);

        vector<Rect> faces;
        detectFaces(resizedFrame, face_cascade_face, faces);

#pragma omp parallel sections
        {
#pragma omp section
            detectEyes(resizedFrame, faces, face_cascade_eye);

#pragma omp section
            detectSmiles(resizedFrame, faces, face_cascade_smile);
        }

        cv::resize(resizedFrame, resizedFrame, Size(frameWidth, frameHeight));
        processedFrames.push_back(resizedFrame.clone());
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "Time: " << elapsed_seconds.count() << endl;

    cap.release();

    VideoWriter video("C:/Users/veras/Desktop/9.04/output.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameWidth, frameHeight));
    for (const auto& frame : processedFrames) {
        video.write(frame);
        imshow("Rezult", frame);

        if (waitKey(10) == 'q') {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}
