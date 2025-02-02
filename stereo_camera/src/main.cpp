#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"
#include "StereoCalibration.hpp"

int main()
{
  // 0. Initialize stereo camera and board
  SingleCamera left_camera("/home/user/calib_data/1204_stereo/Cam_001/");
  SingleCamera right_camera("/home/user/calib_data/1204_stereo/Cam_002/");
  CharucoBoard board(BoardConfig5x5);

  // 1. Calibrate single cameras
  // 1-1. left camera
  cv::Mat K_left, D_left;
  std::vector<cv::Mat> rvecs_left, tvecs_left;
  std::vector<std::vector<cv::Point3f>> allObjPoints3D_left;
  std::vector<std::vector<cv::Point2f>> allImgPoints2D_left;
  std::vector<std::vector<int>> allIds_left;
  double reprojErrLeft = calibrateCamera(left_camera, board, allObjPoints3D_left, allImgPoints2D_left, allIds_left, K_left, D_left, rvecs_left, tvecs_left);
  std::cout << "##### Left Camera's Error: " << reprojErrLeft << " #####"  << std::endl;
  std::cout << "K: " << K_left << std::endl;
  std::cout << "D: " << D_left << std::endl;
  std::cout << "#########################################" << std::endl;
  // 1-2. right camera
  cv::Mat K_right, D_right;
  std::vector<cv::Mat> rvecs_right, tvecs_right;
  std::vector<std::vector<cv::Point3f>> allObjPoints3D_right;
  std::vector<std::vector<cv::Point2f>> allImgPoints2D_right;
  std::vector<std::vector<int>> allIds_right;
  double reprojErrRight = calibrateCamera(right_camera, board, allObjPoints3D_right, allImgPoints2D_right, allIds_right, K_right, D_right, rvecs_right, tvecs_right);
  std::cout << "##### Right Camera's Error: " << reprojErrRight << " #####"  << std::endl;
  std::cout << "K: " << K_right << std::endl;
  std::cout << "D: " << D_right << std::endl;
  std::cout << "#########################################" << std::endl;

  // 2. Find common coners and ids
  // 2D coners
  size_t numImages = left_camera.size();
  // Corners and ids
  std::vector<std::vector<int>> commonIds(numImages);
  std::vector<std::vector<cv::Point2f>> commonCorners_left(numImages);
  std::vector<std::vector<cv::Point2f>> commonCorners_right(numImages);
  for (size_t i = 0; i < numImages; ++i)
  {
    std::vector<int> thisCommonIds;
    std::vector<cv::Point2f> thiscommonCorners_left;
    std::vector<cv::Point2f> thiscommonCorners_right;

    for (size_t leftIdx = 0; leftIdx < allIds_left[i].size(); ++leftIdx)
    {
      int id = allIds_left[i][leftIdx];
      auto it = std::find(allIds_right[i].begin(), allIds_right[i].end(), id);
      if (it != allIds_right[i].end())
      {
        size_t rightIdx = std::distance(allIds_right[i].begin(), it);

        thisCommonIds.push_back(id);
        thiscommonCorners_left.push_back(allImgPoints2D_left[i][leftIdx]);
        thiscommonCorners_right.push_back(allImgPoints2D_right[i][rightIdx]);
      }
    }
    commonIds[i] = thisCommonIds;
    commonCorners_left[i] = thiscommonCorners_left;
    commonCorners_right[i] = thiscommonCorners_right;
  }
  // 3D coners
  std::vector<std::vector<cv::Point3f>> commonCorners3D(numImages);
  for (int i = 0; i < numImages; ++i)
  {
    std::vector<cv::Point3f> temp3D;

    for (size_t k = 0; k < commonIds[i].size(); ++k)
    {
      int cornerId = commonIds[i][k];
      cv::Point3f pt3D = board.getBoard()->chessboardCorners[cornerId];
      temp3D.push_back(pt3D);
    }
    commonCorners3D[i] = temp3D;
  }

  // Visualize common points and projections
  // for (size_t i = 0; i < numImages; i++)
  // {
  //   visualizeCommonCorners(left_camera.getImage(i), right_camera.getImage(i),
  //                          allImgPoints2D_left[i], allImgPoints2D_right[i],
  //                          commonCorners_left[i], commonCorners_right[i]);
  // }

  // 3. Initialize extrinsic paramters
  // Option 1: naive approach
  // Convert to Rodrigues
  std::vector<cv::Mat> R_left(numImages), R_right(numImages);
  for (int i = 0; i < numImages; ++i)
  {
    cv::Rodrigues(rvecs_left[i],  R_left[i]);
    cv::Rodrigues(rvecs_right[i], R_right[i]);
  }
  // Extrinsic at each image
  std::vector<cv::Mat> R_stereo_candidates(numImages), t_stereo_candidates(numImages);
  for (int i = 0; i < numImages; ++i)
  {
    // R_stereo_i = R_right_i * R_left_i^T
    R_stereo_candidates[i] = R_right[i] * R_left[i].t();

    // t_stereo_i = t_right_i - R_stereo_i * t_left_i
    t_stereo_candidates[i] = (tvecs_right[i]) - (R_stereo_candidates[i] * tvecs_left[i]);
  }
  // Rotation average
  cv::Vec3d r_stereo_init(0,0,0);
  for (int i = 0; i < numImages; ++i)
  {
    cv::Mat rvec_temp;
    cv::Rodrigues(R_stereo_candidates[i], rvec_temp); // 3x1
    r_stereo_init += cv::Vec3d(rvec_temp.at<double>(0), rvec_temp.at<double>(1), rvec_temp.at<double>(2));
  }
  r_stereo_init *= (1.0 / numImages); // mean

  cv::Mat R_stereo_init;
  cv::Rodrigues(r_stereo_init, R_stereo_init); // vec -> matrix

  // Translation average
  cv::Mat t_stereo_init = cv::Mat::zeros(3,1,CV_64F);
  for (int i = 0; i < numImages; ++i)
  {
    t_stereo_init += t_stereo_candidates[i];
  }
  t_stereo_init /= (double)numImages;

  // Left -> Right
  std::cout << "Initial R_stereo = \n" << R_stereo_init << std::endl;
  std::cout << "Initial t_stereo = \n" << t_stereo_init << std::endl;

  // Option 2: Epipolar geometry

  // 5. Evaluation
  // w/ stereo BA
  // w/o stereo BA
  // evaluateStereoCalibration(left_camera, right_camera,
  //                           K_left, D_left, K_right, D_right, R_stereo_init, t_stereo_init,
  //                           commonCorners_left, commonCorners_right);

  return 0;
}
