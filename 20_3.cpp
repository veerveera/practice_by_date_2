#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
const int MAX_VALUE = 255;


Mat dorian(Mat& image) {

	Mat dorian_image(image.rows, image.cols, CV_8UC1);

#pragma omp parallel for
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			Vec3b pixel = image.at<Vec3b>(i, j);
			int gray_value = 0.29 * pixel[2] + 0.58 * pixel[1] + 0.11 * pixel[0];
			dorian_image.at<uchar>(i, j) = gray_value;
		}
	}
	return dorian_image;
}


Mat contour(Mat& image) {

	Mat Icontour(image.rows, image.cols, CV_8UC1);
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

#pragma omp parallel for
	for (int i = 1; i < gray.rows - 1; i++) {
		for (int j = 1; j < gray.cols - 1; j++) {
			float gx = gray.at<uchar>(i + 1, j + 1) + 2 * gray.at<uchar>(i, j + 1) + gray.at<uchar>(i - 1, j + 1) - gray.at<uchar>(i + 1, j - 1) - 2 * gray.at<uchar>(i, j - 1) - gray.at<uchar>(i - 1, j - 1);
			float gy = gray.at<uchar>(i + 1, j + 1) + 2 * gray.at<uchar>(i + 1, j) + gray.at<uchar>(i + 1, j - 1) - gray.at<uchar>(i - 1, j - 1) - 2 * gray.at<uchar>(i - 1, j) - gray.at<uchar>(i - 1, j + 1);

			Icontour.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
		}
	}
	return Icontour;
}


Mat negative(Mat& image) {

	Mat Inegative(image.rows, image.cols, CV_8UC3);

#pragma omp parallel for
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {

			Vec3b pixel = image.at<Vec3b>(i, j);
			int blue = pixel[0];
			int green = pixel[1];
			int red = pixel[2];

			Inegative.at<Vec3b>(i, j) = Vec3b(MAX_VALUE - pixel[0], MAX_VALUE - pixel[1], MAX_VALUE - pixel[2]);
		}
	}
	return Inegative;
}


Mat sepia(Mat& image) {

	Mat Isepia(image.rows, image.cols, CV_8UC3);

#pragma omp parallel for
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {

			Vec3b pixel = image.at<Vec3b>(i, j);
			int b = pixel[0];
			int g = pixel[1];
			int r = pixel[2];
			int R = (int)(0.393 * b + 0.769 * g + 0.189 * r);
			int G = (int)(0.349 * b + 0.686 * g + 0.168 * r);
			int B = (int)(0.272 * b + 0.534 * g + 0.131 * r);

			Isepia.at<Vec3b>(i, j) = Vec3b(min(B, MAX_VALUE), min(G, MAX_VALUE), min(R, MAX_VALUE));
		}
	}

	return Isepia;
}

int main() {
	Mat image = imread("C:/Users/veras/Desktop/source_mat.jpg");

	if (image.empty()) {
		cout << "Error!" << endl;
		return -1;
	}

	Mat dorian_image, Icontour, Inegative, Isepia;

#pragma omp parallel sections num_threads(4)
	{
#pragma omp section
		{
			dorian_image = dorian(image);
		}
#pragma omp section
		{
			Icontour = contour(image);

		}
#pragma omp section
		{
			Inegative = negative(image);
		}
#pragma omp section
		{
			Isepia = sepia(image);
		}
	}

	imshow("original", image);
	imshow("gray", dorian_image);
	imshow("contour", Icontour);
	imshow("negative", Inegative);
	imshow("sepia", Isepia);

	waitKey(0);

	return 0;
}
