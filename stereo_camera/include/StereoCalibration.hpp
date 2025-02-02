#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"

double calibrateCamera(SingleCamera& camera,
                       CharucoBoard& board,
                       std::vector<std::vector<cv::Point3f>>& allObjPoints3D,
                       std::vector<std::vector<cv::Point2f>>& allCornersImg,
                       std::vector<std::vector<int>>& allIds,
                       cv::Mat& K_optim,
                       cv::Mat& D_optim,
                       std::vector<cv::Mat>& rvecs,
                       std::vector<cv::Mat>& tvecs)
{
  double image_height = camera.getImage(0).size().height;
  double image_width = camera.getImage(0).size().width;
  cv::Point2f principalPoint;
  principalPoint.x = image_width / 2.0;
  principalPoint.y = image_height / 2.0;
  // Detect and refine corners and ids
  detectAndRefineCorners(camera, board, allCornersImg, allIds);

  // Calibration
  // 1. Generate 3D charuco board points
  allObjPoints3D.resize(allIds.size());
  for (size_t i = 0; i < allIds.size(); i++)
  {
    unsigned int nCorners = (unsigned int)allIds[i].size();
    CV_Assert(nCorners >0 && nCorners == allCornersImg[i].size());
    allObjPoints3D[i].reserve(nCorners);
    for (unsigned int j = 0; j < nCorners; j++)
    {
      int pointId = allIds[i][j];
      CV_Assert(pointId >= 0 && pointId < (int)board.getBoard()->chessboardCorners.size());
      allObjPoints3D[i].push_back(board.getBoard()->chessboardCorners[pointId]);
    }
  }
  // Convert 3D points to 2D points for use convenience
  std::vector<std::vector<cv::Point2f>> allObjPoints2D; // x_w, y_w
  allObjPoints2D.resize(allObjPoints3D.size());
  for (size_t i = 0; i < allObjPoints3D.size(); i++)
  {
    unsigned int nPoints = (unsigned int)allObjPoints3D[i].size();
    allObjPoints2D[i].reserve(nPoints);
    for (unsigned int j = 0; j < nPoints; j++)
    {
      cv::Point3f point3D = allObjPoints3D[i][j];
      allObjPoints2D[i].emplace_back(point3D.x, point3D.y);
    }
  }

  // 2. Find Homography for each image
  std::vector<cv::Mat> homographies;
  cv::Mat normalzationMatObj;
  cv::Mat normalzationMatImg;
  for(size_t i = 0; i < allCornersImg.size(); i++)
  {
    // 2-1. Normalize the object and image points and find homography(normalized)
    std::vector<cv::Point2f> normalizedObjPoints;
    std::vector<cv::Point2f> normalizedImgPoints;
    normalzationMatObj = calculateNormalizationMat(allObjPoints2D[i]);
    normalzationMatImg = calculateNormalizationMat(allCornersImg[i]);
    normalizePoints(allObjPoints2D[i], normalizedObjPoints, normalzationMatObj);
    normalizePoints(allCornersImg[i], normalizedImgPoints, normalzationMatImg);
    // 2-2. Find initial homography(normalized)
    // cv::Mat H_normal = findHomographyForEachImage(normalizedObjPoints, normalizedImgPoints); //FIXME
    cv::Mat H_normal = cv::findHomography(normalizedObjPoints, normalizedImgPoints, cv::RANSAC);
    // 2-3. Denormalize the homography
    cv::Mat H = normalzationMatImg.inv() * H_normal * normalzationMatObj;
    // 2-4. Optimize the homography
    H = optimizeHomography(H, allObjPoints2D[i], allCornersImg[i]);
    H /= H.at<double>(2, 2); // for h33 = 1
    //visualizeHomographyProjection(camera.getImage(i), allObjPoints3D[i], allCornersImg[i], H);
    homographies.push_back(H);
  }
  // 3. Estimate initial intrinsic matrix
  cv::Mat K_init = estimateInitialIntrinsicLLT(homographies);

  // 4. Estimate initial extrinsic matrix
  std::vector<cv::Mat> rotationMatrices;
  std::vector<cv::Mat> translationVectors;
  initializeExtrinsic(homographies, K_init, rotationMatrices, translationVectors);

  // 5. Estimate radial lens distortion
  cv::Mat D_init;
  initializeDistortion(allObjPoints2D, allCornersImg, homographies, principalPoint, D_init);

  // 6. Optimize total projection error
  ceres::Problem problem;
  double K[5] = {K_init.at<double>(0, 0), K_init.at<double>(1, 1), K_init.at<double>(0, 2), K_init.at<double>(1, 2), K_init.at<double>(0, 1)};
  double D[5] = {D_init.at<double>(0, 0), D_init.at<double>(1, 0), D_init.at<double>(2, 0), D_init.at<double>(3, 0), D_init.at<double>(4, 0)};

  // Add K and D to parameter
  problem.AddParameterBlock(K, 5);  // [fx, fy, cx, cy, s]
  problem.AddParameterBlock(D, 5);  // [k1, k2, p1, p2, k3]

  std::vector<std::array<double, 3>> all_rvecs(allObjPoints3D.size());
  std::vector<std::array<double, 3>> all_tvecs(allObjPoints3D.size());

  for (size_t i = 0; i < allObjPoints3D.size(); ++i) {
    cv::Rodrigues(rotationMatrices[i], cv::Mat(3, 1, CV_64F, all_rvecs[i].data()));
    memcpy(all_tvecs[i].data(), translationVectors[i].ptr<double>(), 3 * sizeof(double));

    // Add R and t per image to parameter blocks
    problem.AddParameterBlock(all_rvecs[i].data(), 3);
    problem.AddParameterBlock(all_tvecs[i].data(), 3);

    for (size_t j = 0; j < allObjPoints3D[i].size(); ++j) {
      cv::Point3f obj = allObjPoints3D[i][j];
      cv::Point2f img = allCornersImg[i][j];

      ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

      // Residual
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CalibrationReprojectionError, 2, 3, 3, 5, 5>(
            new CalibrationReprojectionError(obj.x, obj.y, img.x, img.y)),
        loss_function,
        all_rvecs[i].data(),  // rvec on image[i]
        all_tvecs[i].data(),  // tvec on image[i]
        K,                    // K
        D                     // D
      );
    }
  }

  // Solver Option
  ceres::Solver::Options options;
  // Levenberg-Marquardt
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.initial_trust_region_radius = 1e3;
  options.min_lm_diagonal = 1e-6;
  options.max_lm_diagonal = 1e6;
  options.max_num_iterations = 500;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::PER_MINIMIZER_ITERATION;

  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  convertVecArray2VecCVMat(all_rvecs, rvecs);
  convertVecArray2VecCVMat(all_tvecs, tvecs);
  K_optim = (cv::Mat_<double>(3, 3) << K[0], K[4], K[2],
                                        0.0, K[1], K[3],
                                        0.0, 0.0, 1.0);
  D_optim = cv::Mat(1, 5, CV_64F, D).clone();

  // evaluate
  double reprojection_err;
  reprojection_err = calculateReprojectionError(allObjPoints3D, allCornersImg, rvecs, tvecs, K_optim, D_optim);

  return reprojection_err;
}

struct Point2fCompare
{
  bool operator()(const cv::Point2f& a, const cv::Point2f& b) const
  {
    return (a.x != b.x || a.y != b.y);
  }
};

void visualizeCommonCorners(cv::Mat image_left, cv::Mat image_right,
                            std::vector<cv::Point2f>& corners_left, std::vector<cv::Point2f>& corners_right,
                            std::vector<cv::Point2f>& corners_common_left, std::vector<cv::Point2f>& corners_common_right)
{
  cv::Mat vis_left = image_left.clone();
  cv::Mat vis_right = image_right.clone();

  std::vector<cv::Point2f> unique_left, unique_right;
  std::set<cv::Point2f, Point2fCompare> common_left_set(corners_common_left.begin(), corners_common_left.end());
  std::set<cv::Point2f, Point2fCompare> common_right_set(corners_common_right.begin(), corners_common_right.end());

  for (const auto& pt : corners_left)
  {
    if (common_left_set.find(pt) == common_left_set.end())
      unique_left.push_back(pt);
  }
  for (const auto& pt : corners_right)
  {
    if (common_right_set.find(pt) == common_right_set.end())
      unique_right.push_back(pt);
  }

  for (const auto& pt : unique_left)
    cv::circle(vis_left, pt, 5, cv::Scalar(255,0,0), -1);
  for (const auto& pt : unique_right)
    cv::circle(vis_right, pt, 5, cv::Scalar(0,255,0), -1);
  for (const auto& pt : corners_common_left)
    cv::circle(vis_left, pt, 5, cv::Scalar(0,255,0), -1);
  for (const auto& pt : corners_common_right)
    cv::circle(vis_right, pt, 5, cv::Scalar(0,255,0), -1);

  cv::Mat concat;
  cv::hconcat(vis_left, vis_right, concat);

  cv::imshow("Common Corners", concat);
  cv::waitKey(0);
}



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

void evaluateStereoCalibration(SingleCamera& left_camera, SingleCamera& right_camera,
                               cv::Mat& K_left, cv::Mat& D_left, cv::Mat& K_right, cv::Mat& D_right, cv::Mat& R_stereo, cv::Mat& t_stereo,
                               std::vector<std::vector<cv::Point2f>> commonCorners_left, std::vector<std::vector<cv::Point2f>> commonCorners_right)
{
  cv::Size imageSize(left_camera.getImage(0).cols, left_camera.getImage(0).rows);
  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(
    K_left,  D_left,
    K_right, D_right,
    imageSize,
    R_stereo, t_stereo,
    R1, R2, P1, P2, Q,
    0,
    0.0, // alpha
    imageSize
  );
  // Y-axis error
  double yAxisError = computeVerticalErrorRMS(commonCorners_left, commonCorners_right,
                                              K_left,  D_left,  R1, P1,
                                              K_right, D_right, R2, P2);

  std::cout << "Y-axis error(RMSE): " << yAxisError << std::endl;

  // Create remap maps for each camera
  cv::Mat map1L, map2L, map1R, map2R;
  cv::initUndistortRectifyMap(
    K_left, D_left, R1, P1,
    imageSize, CV_32FC1,
    map1L, map2L
  );
  cv::initUndistortRectifyMap(
    K_right, D_right, R2, P2,
    imageSize, CV_32FC1,
    map1R, map2R
  );

  // Remap the original images -> rectified images
  size_t numImages = left_camera.size();
  std::vector<cv::Mat> leftRect(numImages), rightRect(numImages);
  for (int i = 0; i < numImages; i++)
  {
    cv::remap(left_camera.getImage(i),  leftRect[i],  map1L, map2L, cv::INTER_LINEAR);
    cv::remap(right_camera.getImage(i), rightRect[i], map1R, map2R, cv::INTER_LINEAR);

    cv::Mat sideBySide;
    cv::hconcat(leftRect[i], rightRect[i], sideBySide);

    int step = 40;
    for (int y = 0; y < sideBySide.rows; y += step)
    {
      cv::line(sideBySide,
              cv::Point(0, y),
              cv::Point(sideBySide.cols - 1, y),
              cv::Scalar(0, 255, 0),
              1);
    }

    cv::imshow("Rectified Pair with horizontal lines", sideBySide);
    cv::waitKey(0);
  }
}
