//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C
#include <stdio.h>
//C++
#include <fstream>
#include <iostream>
#include <sstream>
using cv::Mat;


#define IM_W 640
#define IM_H 480

// Global variables
Mat frame; //current frame

int keyboard; //input from keyboard





void help();
void processVideo();
using namespace cv;
cv::VideoCapture capture;

int main(int argc, char* argv[])
{
    capture=cv::VideoCapture(0);
    if(!capture.isOpened())
    {
        std::cerr<<"Failed to open Camera"<<std::endl;
        exit(1);
    }


   
    processVideo();
    //destroy GUI windows
    cv::destroyAllWindows();
}

cv::Point polygon_center(std::vector<cv::Point> poly){
    int sx = 0, sy = 0;
    for(auto p : poly){
        sx += p.x;
        sy += p.y;
    }
    return cv::Point(sx/poly.size(), sy/poly.size());
}

std::vector<cv::Point> create_sample_points(int n_x, int n_y, int size, cv::Point start_point, int off_x, int off_y, Mat &frame){
    std::vector<Point> points_capture;

    for(int i = 0; i < n_x; i += 1){
        for(int j = 0; j < n_y; j+= 1){
            Point p1 = start_point + Point(i * off_x, j * off_y);
            points_capture.push_back(p1);
            Point p2 = p1 + Point(size, size);
            rectangle(frame, p1, p2, Scalar(0, 0, 255));
        }
    }
    return points_capture;
}

int get_biggest_label_id(Mat connect_stats, int label_nb){
    int biggest_label = 1, max_area = connect_stats.at<int>(1, CC_STAT_AREA);
    for(int label = 2; label < label_nb; label++) {
        int area = connect_stats.at<int32_t>(label, CC_STAT_AREA);
        if (area > max_area) {
            biggest_label = label;
            max_area = area;
        }
    }
    return biggest_label;
}

void draw_nth_label(Mat &frame, Mat &masque, int n_label, Mat labels){
    for(int j = 0; j < IM_W; j++){
        for(int i = 0; i < IM_H; i++){
            int label = labels.at<int32_t>(i, j);
            if(label == n_label)frame.at<Vec3b>(i, j) = {128, 0, 0};
            if(label == n_label)masque.at<char>(i, j) = 255;
            else masque.at<char>(i, j) = 0;
        }
    }
}


void binarize_image(std::vector<cv::Mat> subs, Mat &masque_resultant, int delta_h, int delta_ls){
    int i = 0;
    Mat frame_hls = Mat(IM_H, IM_W, CV_8UC3);
    cv::cvtColor(frame, frame_hls, CV_BGR2HLS);
    Mat masque_temp;

    for(auto m : subs) {
        Mat hls_m = Mat(10, 10, CV_8UC3);
        cv::cvtColor(m, hls_m, CV_BGR2HLS);
        Vec3b hls = hls_m.at<Vec3b>(5, 5);
        cv::inRange(frame_hls, cv::Scalar(hls[0] - delta_h, hls[1] - delta_ls, hls[2] - delta_ls),
                cv::Scalar(hls[0] + delta_h, hls[1] + delta_ls, hls[2] + delta_ls), masque_temp);
        if (i == 0) masque_resultant = masque_temp.clone();
        else masque_resultant += masque_temp;
        i++;
    }
}
std::vector<cv::Mat> capture_subdivisions(cv::Point taille_captures, cv::Mat frame, std::vector<cv::Point> points_capture){
    std::vector<cv::Mat> subs;
    for(auto p : points_capture){
        Mat sample = Mat(frame, cv::Rect(p, p + taille_captures)).clone();
        medianBlur ( sample, sample, 15);
        subs.push_back(sample);
    }
    return subs;
}

cv::Point get_hand_pos(Mat &frame, std::vector<Mat> hand_subdivisions) {
    int label_nb;
    medianBlur ( frame, frame, 9);
    Mat labels, connect_stats, centroids, mask = Mat(IM_H, IM_W, CV_8U);
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    binarize_image(hand_subdivisions, mask, 5, 10);
    cv::Mat strel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(9,9));
    cv::dilate(mask, mask, strel);
    label_nb = cv::connectedComponentsWithStats(mask, labels, connect_stats, centroids, 8, CV_32S);


    Mat objet = Mat(IM_H, IM_W, CV_8UC1);
    objet = 0;
    int biggest_label = get_biggest_label_id(connect_stats, label_nb);
    draw_nth_label(frame, objet, biggest_label, labels);
    cv::findContours( objet, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    std::vector<std::vector<Point> >hull( 1 );
    convexHull( Mat(contours[0]), hull[0], false );
    drawContours( frame, hull, 0, {0, 255, 0}, 4, 8, std::vector<Vec4i>(), 0, Point() );
    cv::Point centre = (polygon_center(hull[0]));
    cv::circle(frame, centre, 3, cv::Scalar(0, 0, 255), 3);
    return centre;
}

void processVideo(){

    std::vector<Point> points_capture;
    std::vector<Mat> hand_subdivisions;

    bool hand_captured = false;

    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        if(!capture.read(frame)) {
            std::cout << "Failed to read from webcam." << std::endl;
            exit(EXIT_FAILURE);
        }
        keyboard = waitKey( 30 );


        if(!hand_captured)
            points_capture = create_sample_points(4, 7, 10, cv::Point(100, 200), 10, 10, frame);
        else {
            Mat frame_clone = frame.clone();
            cv::Point centre = get_hand_pos(frame_clone, hand_subdivisions);
            cv::circle(frame, centre, 3, cv::Scalar(0, 0, 255), 3);
        }

        if((char)keyboard == 's' && !hand_captured){
            hand_subdivisions = capture_subdivisions(cv::Point(10, 10), frame, points_capture);
            hand_captured = true;
        }

        imshow("frame", frame);
    }
    capture.release();
}
