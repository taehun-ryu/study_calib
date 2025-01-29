#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

double computeVerticalErrorRMS(
  const std::vector<std::vector<cv::Point2f>>& commonCornersLeft,
  const std::vector<std::vector<cv::Point2f>>& commonCornersRight,
  const cv::Mat& K_left, const cv::Mat& D_left,
  const cv::Mat& R1, const cv::Mat& P1,
  const cv::Mat& K_right, const cv::Mat& D_right,
  const cv::Mat& R2, const cv::Mat& P2)
{
  double totalSquareError = 0.0;
  int totalPoints = 0;
  for (size_t i = 0; i < commonCornersLeft.size(); i++) {
    std::vector<cv::Point2f> undistortLeft, undistortRight;
    // convert to 'rectified' coordinates
    cv::undistortPoints(commonCornersLeft[i], undistortLeft, K_left, D_left, R1, P1);
    cv::undistortPoints(commonCornersRight[i], undistortRight, K_right, D_right, R2, P2);
    for (size_t j = 0; j < undistortLeft.size(); j++) {
      double yL = undistortLeft[j].y;
      double yR = undistortRight[j].y;
      double dy = yL - yR;
      totalSquareError += (dy * dy);
      totalPoints++;
    }
  }
  double rmsVerticalError = 0.0;
  if (totalPoints > 0) {
    rmsVerticalError = std::sqrt(totalSquareError / totalPoints);
  }
  return rmsVerticalError;
}
