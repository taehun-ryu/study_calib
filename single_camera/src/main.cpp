#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"


int main()
{
  // Initialize SingleCamera and CharucoBoard
  SingleCamera camera("/home/user/calib_data/1204_stereo/Cam_002/");
  double image_height = camera.getImage(0).size().height;
  double image_width = camera.getImage(0).size().width;
  cv::Point2f principalPoint;
  principalPoint.x = image_width / 2.0;
  principalPoint.y = image_height / 2.0;

  CharucoBoard board(BoardConfig5x5);
  //board.showCharucoBoard("charuco board");

  // Detect and refine corners and ids
  std::vector<std::vector<cv::Point2f>> allCornersImg; // x_c, y_c
  std::vector<std::vector<int>> allIds;
  detectAndRefineCorners(camera, board, allCornersImg, allIds);

  // Calibration
  // 1. Generate 3D charuco board points
  std::vector<std::vector<cv::Point3f>> allObjPoints3D; // x_w, y_w, 0
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
    cv::Mat H_normal = findHomographyForEachImage(normalizedObjPoints, normalizedImgPoints);
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
  std::cout << "K_init: " << std::endl << K_init << std::endl;

  // 4. Estimate initial extrinsic matrix
  std::vector<cv::Mat> rotationMatrices;
  std::vector<cv::Mat> translationVectors;
  initializeExtrinsic(homographies, K_init, rotationMatrices, translationVectors);

  // 5. Estimate radial lens distortion
  cv::Mat D_init;
  initializeDistortion(allObjPoints2D, allCornersImg, homographies, principalPoint, D_init);
  std::cout << "D_init :" << std::endl;
  std::cout << "  k1: " << D_init.at<double>(0,0) << std::endl;
  std::cout << "  k2: " << D_init.at<double>(1,0) << std::endl;
  std::cout << "  p1: " << D_init.at<double>(2,0) << std::endl;
  std::cout << "  p2: " << D_init.at<double>(3,0) << std::endl;
  std::cout << "  k3: " << D_init.at<double>(4,0) << std::endl;

  // 6. Optimize total projection error
  ceres::Problem problem;
#if SKEW_COEFFICIENT
  double K[5] = {K_init.at<double>(0, 0), K_init.at<double>(1, 1), K_init.at<double>(0, 2), K_init.at<double>(1, 2), K_init.at<double>(0, 1)};
#else
  double K[4] = {K_init.at<double>(0, 0), K_init.at<double>(1, 1), K_init.at<double>(0, 2), K_init.at<double>(1, 2)};
#endif
  double D[5] = {D_init.at<double>(0, 0), D_init.at<double>(1, 0), D_init.at<double>(2, 0), D_init.at<double>(3, 0), D_init.at<double>(4, 0)};

  // Add K and D to parameter
#if SKEW_COEFFICIENT
  problem.AddParameterBlock(K, 5);  // [fx, fy, cx, cy, s]
#else
  problem.AddParameterBlock(K, 4);
#endif
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
#if SKEW_COEFFICIENT
        new ceres::AutoDiffCostFunction<CalibrationReprojectionError, 2, 3, 3, 5, 5>(
            new CalibrationReprojectionError(obj.x, obj.y, img.x, img.y)),
#else
        new ceres::AutoDiffCostFunction<CalibrationReprojectionError, 2, 3, 3, 4, 5>(
            new CalibrationReprojectionError(obj.x, obj.y, img.x, img.y)),
#endif
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
  std::cout << summary.FullReport() << std::endl;

  std::vector<cv::Mat> rvecs, tvecs;
  convertVecArray2VecCVMat(all_rvecs, rvecs);
  convertVecArray2VecCVMat(all_tvecs, tvecs);
#if SKEW_COEFFICIENT
  cv::Mat K_optim = (cv::Mat_<double>(3, 3) << K[0], K[4], K[2],
                                               0.0, K[1], K[3],
                                               0.0, 0.0, 1.0);
#else
  cv::Mat K_optim = (cv::Mat_<double>(3, 3) << K[0], 0, K[2],
                                               0.0, K[1], K[3],
                                               0.0, 0.0, 1.0);
#endif
  cv::Mat D_optim = cv::Mat(1, 5, CV_64F, D).clone();
  std::cout << "Optimized K: " << K_optim << std::endl;
  std::cout << "Optimized D: " << D_optim << std::endl;

  // 7. Evaluate optimization result
  double reprojection_err;
  reprojection_err = calculateReprojectionError(allObjPoints3D, allCornersImg, rvecs, tvecs, K_optim, D_optim);
  std::cout << "Reprojection Error(RMSE): " << reprojection_err << std::endl;
  for (int i = 0; i < 1; i++)
  {
    validateDistortionCorrection(camera.getImage(i), K_optim, D_optim);
    visualizeReprojection(camera.getImage(i), allObjPoints3D[i], allCornersImg[i], rvecs[i], tvecs[i], K_optim, D_optim);
  }


  return 0;
}
