#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"
#include "StereoCalibration.hpp"

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
  cv::Point2f principalPoint(camera.getImage(0).size().width, camera.getImage(0).size().height);
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

  // 4. Estimate initial extrinsic matrix
  std::vector<cv::Mat> rotationMatrices;
  std::vector<cv::Mat> translationVectors;
  initializeExtrinsic(homographies, K_init, rotationMatrices, translationVectors);

  // 5. Estimate radial lens distortion
  cv::Mat D_init;
  initializeDistortion(allObjPoints2D, allCornersImg, homographies, principalPoint, D_init);

  // 6. Optimize total projection error
  ceres::Problem problem;
  double K[5] = {K_init.at<double>(0, 0), K_init.at<double>(1, 1), K_init.at<double>(0, 2), K_init.at<double>(1, 2)};
  double D[5] = {D_init.at<double>(0, 0), D_init.at<double>(1, 0), D_init.at<double>(2, 0), D_init.at<double>(3, 0), D_init.at<double>(4, 0)};

  // Add K and D to parameter
  problem.AddParameterBlock(K, 4);  // [fx, fy, cx, cy]
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
        new ceres::AutoDiffCostFunction<CalibrationReprojectionError, 2, 3, 3, 4, 5>(
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
  options.initial_trust_region_radius = 1e10;
  options.min_lm_diagonal = 1e-2;
  options.max_lm_diagonal = 1e32;
  options.max_num_iterations = 200;
  options.minimizer_progress_to_stdout = false;

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

int main()
{
  // 0. Initialize stereo camera and board
  SingleCamera left_camera("/home/ryu/calib_data/2025-w/1204_stereo/Cam_001/");
  SingleCamera right_camera("/home/ryu/calib_data/2025-w/1204_stereo/Cam_002/");
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
  int numImages = left_camera.size();
  std::vector<std::vector<int>> commonIds(numImages);
  std::vector<std::vector<cv::Point2f>> commonCorners_left(numImages);
  std::vector<std::vector<cv::Point2f>> commonCorners_right(numImages);
  for (int i = 0; i < numImages; ++i)
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

  // 3. Initialize extrinsic paramters
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

  std::cout << "Initial R_stereo = \n" << R_stereo_init << std::endl;
  std::cout << "Initial t_stereo = \n" << t_stereo_init << std::endl;

  // 4. Run stereo BA
  // TODO: optimization 구현

  // 5. Evaluation
  cv::Mat R1, R2, P1, P2, Q;
  cv::Size imageSize(left_camera.getImage(0).cols, left_camera.getImage(0).rows);
  cv::stereoRectify(
    K_left,  D_left,
    K_right, D_right,
    imageSize,
    R_stereo_init, t_stereo_init,
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

  return 0;
}
