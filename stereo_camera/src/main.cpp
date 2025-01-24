#include <iostream>
#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"

int main()
{
  // 0. Initialize stereo camera and board
  // left camera
  SingleCamera cameraLeft("/home/user/calib_data/1204_stereo/Cam_001/");
  cv::Point2f principalPointLeft(cameraLeft.getImage(0).size().width, cameraLeft.getImage(0).size().height);
  // right camera
  SingleCamera cameraRight("/home/user/calib_data/1204_stereo/Cam_002/");
  cv::Point2f principalPointRight(cameraRight.getImage(0).size().width, cameraRight.getImage(0).size().height);
  // board
  CharucoBoard board(BoardConfig5x5);

  // 1. Detect charuco board corner

  return 0;
}
