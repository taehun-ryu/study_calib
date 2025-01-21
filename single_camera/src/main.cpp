#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"


int main() {
  // Initialize SingleCamera and CharucoBoard
  SingleCamera camera("/home/user/calib_data/1204_stereo/Cam_002/");
  cv::Point2f principalPoint(camera.getImage(0).size().width, camera.getImage(0).size().height);

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
    //visualizeProjection(camera.getImage(i), allObjPoints3D[i], allCornersImg[i], H);
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
  double K[5] = {K_init.at<double>(0, 0), K_init.at<double>(1, 1), K_init.at<double>(0, 2), K_init.at<double>(1, 2)};
  double D[5] = {D_init.at<double>(0, 0), D_init.at<double>(1, 0), D_init.at<double>(2, 0), D_init.at<double>(3, 0), D_init.at<double>(4, 0)};

  for (size_t i = 0; i < allObjPoints3D.size(); ++i) {
    double rvec[3], tvec[3];
    cv::Rodrigues(rotationMatrices[i], cv::Mat(3, 1, CV_64F, rvec));
    memcpy(tvec, translationVectors[i].ptr<double>(), 3 * sizeof(double));

    for (size_t j = 0; j < allObjPoints3D[i].size(); ++j) {
      cv::Point3f obj = allObjPoints3D[i][j];
      cv::Point2f img = allCornersImg[i][j];

      ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CalibrationReprojectionError, 2, 3, 3, 4, 5>(
            new CalibrationReprojectionError(obj.x, obj.y, img.x, img.y)),
        loss_function,  // 손실 함수 추가
        rvec, tvec, K, D);

    }
  }

  // Solver 옵션 설정
  ceres::Solver::Options options;
  // Levenberg-Marquardt
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  //options.trust_region_strategy_type = ceres::DOGLEG;
  options.initial_trust_region_radius = 1e10;
  options.min_lm_diagonal = 1e-2;
  options.max_lm_diagonal = 1e32;
  options.max_num_iterations = 200;
  options.minimizer_progress_to_stdout = false;

  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // 결과 출력
  std::cout << summary.FullReport() << "\n";
  std::cout << "Optimized K: fx=" << K[0] << ", fy=" << K[1]
            << ", cx=" << K[2] << ", cy=" << K[3] << ", s=" << K[4] << "\n";
  std::cout << "Optimized D: k1=" << D[0] << ", k2=" << D[1]
            << ", p1=" << D[2] << ", p2=" << D[3] << ", k3=" << D[4] << "\n";

  return 0;
}
