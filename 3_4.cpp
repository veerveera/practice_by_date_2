#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace cv;
using namespace std;

int mandelbrot(double cr, double ci, int max_iterations) {

    double zr = 0.0, zi = 0.0;
    int iterations = 0;

    while (zr * zr + zi * zi <= 4.0 && iterations < max_iterations) {
        double temp = zr * zr - zi * zi + cr;
        zi = 2 * zr * zi + ci;
        zr = temp;
        iterations++;
    }

    return iterations;
}

Vec3b color(int iterations) {
    if (iterations >= 255) {
        return Vec3b(0, 0, 0);
    }
    else { 
        return Vec3b((iterations % 11 + 1) / 9.0 * 255, (iterations % 11 + 1) / 5.0 * 255, (iterations % 19 + 1) / 7.0 * 255);
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 1920;
    int height = 1080;
    int max_iterations = 992;
    int start = rank * (height / size);
    int end = min((rank + 1) * (height / size), height);

    Mat emae(end - start, width, CV_8UC3, Scalar(0, 0, 0));

    for (int j = start; j < end; j++) {
        for (int i = 0; i < width; i++) {
            double x = -2.5 + (2.5 - (-2.5)) * i / width;
            double y = -1.5 + (1.5 - (-1.5)) * j / height;

            int iterations = mandelbrot(x, y, max_iterations);

            emae.at<Vec3b>(j - start, i) = color(iterations);
        }
    }

    Mat emae_itog(height, width, CV_8UC3);

    MPI_Gather(emae.data, (end - start) * width * 3, MPI_UNSIGNED_CHAR, emae_itog.data, (end - start) * width * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        imshow("Mandelbrot", emae_itog);
        waitKey(0);
    }
    MPI_Finalize();
    return 0;
}
