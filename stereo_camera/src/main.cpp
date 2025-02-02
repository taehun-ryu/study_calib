#include "Camera.hpp"
#include "CharucoBoard.hpp"
#include "SingleCalibration.hpp"
#include "StereoCalibration.hpp"

#define IS_INTRINSIC_FIXED false

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
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
  cv::Mat R_stereo_init = averageRotationMatrix(R_stereo_candidates);

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

  // 4. Stereo BA
  ceres::Problem problem;

  std::vector<std::array<double, 3>> all_rvecs_left(commonCorners3D.size());
  std::vector<std::array<double, 3>> all_tvecs_left(commonCorners3D.size());
  std::vector<std::array<double, 3>> all_rvecs_right(commonCorners3D.size());
  std::vector<std::array<double, 3>> all_tvecs_right(commonCorners3D.size());

  // Add intrinsic and distortion to parameter block
#if SKEW_COEFFICIENT
  double K_l[5] = {K_left.at<double>(0, 0), K_left.at<double>(1, 1), K_left.at<double>(0, 2), K_left.at<double>(1, 2), K_left.at<double>(0, 1)};
#else
  double K_l[4] = {K_left.at<double>(0, 0), K_left.at<double>(1, 1), K_left.at<double>(0, 2), K_left.at<double>(1, 2)};
#endif

  double d_l[5] = {D_left.at<double>(0, 0), D_left.at<double>(0, 1), D_left.at<double>(0, 2), D_left.at<double>(0, 3), D_left.at<double>(0, 4)};

#if SKEW_COEFFICIENT
  double K_r[5] = {K_right.at<double>(0, 0), K_right.at<double>(1, 1), K_right.at<double>(0, 2), K_right.at<double>(1, 2), K_right.at<double>(0, 1)};
#else
  double K_r[4] = {K_right.at<double>(0, 0), K_right.at<double>(1, 1), K_right.at<double>(0, 2), K_right.at<double>(1, 2)};
#endif

  double d_r[5] = {D_right.at<double>(0, 0), D_right.at<double>(0, 1), D_right.at<double>(0, 2), D_right.at<double>(0, 3), D_right.at<double>(0, 4)};

#if SKEW_COEFFICIENT
  problem.AddParameterBlock(K_l, 5);  // [fx, fy, cx, cy, s]
#else
  problem.AddParameterBlock(K_l, 4);
#endif
  problem.AddParameterBlock(d_l, 5);  // [k1, k2, p1, p2, k3]

#if SKEW_COEFFICIENT
  problem.AddParameterBlock(K_r, 5);
#else
  problem.AddParameterBlock(K_r, 4);
#endif
  problem.AddParameterBlock(d_r, 5);

  if (IS_INTRINSIC_FIXED)
  {
    problem.SetParameterBlockConstant(K_l);
    problem.SetParameterBlockConstant(d_l);
    problem.SetParameterBlockConstant(K_r);
    problem.SetParameterBlockConstant(d_r);
  }
  else
  {
    problem.SetParameterBlockVariable(K_l);
    problem.SetParameterBlockVariable(d_l);
    problem.SetParameterBlockVariable(K_r);
    problem.SetParameterBlockVariable(d_r);
#if SKEW_COEFFICIENT
    problem.SetParameterLowerBound(K_l, 4, -0.1);
    problem.SetParameterUpperBound(K_l, 4, 0.1);
    problem.SetParameterLowerBound(K_r, 4, -0.1);
    problem.SetParameterUpperBound(K_r, 4, 0.1);
#endif
  }

  //Add extrinsic to parameter block
  cv::Mat rvec_stereo_init;
  cv::Rodrigues(R_stereo_init, rvec_stereo_init);
  double rvec_init[3] = {rvec_stereo_init.at<double>(0,0), rvec_stereo_init.at<double>(1,0), rvec_stereo_init.at<double>(2,0)};
  double tvec_init[3] = {t_stereo_init.at<double>(0,0), t_stereo_init.at<double>(1,0), t_stereo_init.at<double>(2,0)};

  problem.AddParameterBlock(rvec_init, 3);
  problem.AddParameterBlock(tvec_init, 3);

  for (size_t i = 0; i < commonCorners3D.size(); i++)
  {
    cv::Rodrigues(R_left[i], cv::Mat(3, 1, CV_64F, all_rvecs_left[i].data()));
    memcpy(all_tvecs_left[i].data(), tvecs_left[i].ptr<double>(), 3 * sizeof(double));
    cv::Rodrigues(R_right[i], cv::Mat(3, 1, CV_64F, all_rvecs_right[i].data()));
    memcpy(all_tvecs_right[i].data(), tvecs_right[i].ptr<double>(), 3 * sizeof(double));

    // Add R and t per image to parameter blocks
    problem.AddParameterBlock(all_rvecs_left[i].data(), 3);
    problem.AddParameterBlock(all_tvecs_left[i].data(), 3);
    problem.AddParameterBlock(all_rvecs_right[i].data(), 3);
    problem.AddParameterBlock(all_tvecs_right[i].data(), 3);

    if (IS_INTRINSIC_FIXED)
    {
      problem.SetParameterBlockConstant(all_rvecs_left[i].data());
      problem.SetParameterBlockConstant(all_tvecs_left[i].data());
      problem.SetParameterBlockConstant(all_rvecs_right[i].data());
      problem.SetParameterBlockConstant(all_tvecs_right[i].data());
    }
    else
    {
      problem.SetParameterBlockVariable(all_rvecs_left[i].data());
      problem.SetParameterBlockVariable(all_tvecs_left[i].data());
      problem.SetParameterBlockVariable(all_rvecs_right[i].data());
      problem.SetParameterBlockVariable(all_tvecs_right[i].data());
    }

    for (size_t j = 0; j < commonCorners3D[i].size(); j++)
    {
      cv::Point3f obj = commonCorners3D[i][j];
      cv::Point2f left_img = commonCorners_left[i][j];
      cv::Point2f right_img = commonCorners_right[i][j];

      ceres::LossFunction* huber_loss_function = new ceres::HuberLoss(0.1);

      problem.AddResidualBlock( // Add stereo reprojection risidual
        // Cost function
        StereoReprojectionResidual::Create(obj.x, obj.y, left_img.x, left_img.y, right_img.x, right_img.y),
        // loss function
        huber_loss_function,
        // Optimization values: 10
        K_l,
        d_l,
        K_r,
        d_r,
        all_rvecs_left[i].data(),
        all_tvecs_left[i].data(),
        all_rvecs_right[i].data(),
        all_tvecs_right[i].data(),
        rvec_init,
        tvec_init
      );
      problem.AddResidualBlock( // Add Epipolar Constraint
        StereoEpipolarResidual::Create(left_img.x, left_img.y, right_img.x, right_img.y),
        huber_loss_function,
        K_l,
        K_r,
        rvec_init,
        tvec_init
      );
    }
  }

  // Option
  ceres::Solver::Options options;
  // Common
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 4;
  options.dense_linear_algebra_library_type = ceres::CUDA;
  // should test
  options.initial_trust_region_radius = 1e3;
  options.max_trust_region_radius = 1e10;
  options.min_lm_diagonal = 1e-6;
  options.max_lm_diagonal = 1e6;
  options.max_num_iterations = 500;
  options.use_nonmonotonic_steps = true;

  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  // Optimization Results
  // Intrinsic - Left camera
#if SKEW_COEFFICIENT
  cv::Mat K_left_optim = (cv::Mat_<double>(3, 3) << K_l[0], K_l[4], K_l[2],
                                               0.0, K_l[1], K_l[3],
                                               0.0, 0.0, 1.0);
#else
  cv::Mat K_left_optim = (cv::Mat_<double>(3, 3) << K_l[0], 0, K_l[2],
                                               0.0, K_l[1], K_l[3],
                                               0.0, 0.0, 1.0);
#endif
  cv::Mat D_left_optim = cv::Mat(1, 5, CV_64F, d_l).clone();
  std::cout << "Optimized left - K: " << K_left_optim << std::endl;
  std::cout << "Optimized left - D: " << D_left_optim << std::endl;

  // Intrinsic - Right camera
#if SKEW_COEFFICIENT
  cv::Mat K_right_optim = (cv::Mat_<double>(3, 3) << K_r[0], K_r[4], K_r[2],
                                              0.0, K_r[1], K_r[3],
                                              0.0, 0.0, 1.0);
#else
  cv::Mat K_right_optim = (cv::Mat_<double>(3, 3) << K_r[0], 0.0, K_r[2],
                                              0.0, K_r[1], K_r[3],
                                              0.0, 0.0, 1.0);
#endif
  cv::Mat D_right_optim = cv::Mat(1, 5, CV_64F, d_r).clone();
  std::cout << "Optimized right - K: " << K_right_optim << std::endl;
  std::cout << "Optimized right - D: " << D_right_optim << std::endl;

  cv::Mat r_stereo = (cv::Mat_<double>(3, 1) << rvec_init[0], rvec_init[1], rvec_init[2]);
  cv::Mat t_stereo = (cv::Mat_<double>(3, 1) << tvec_init[0], tvec_init[1], tvec_init[2]);
  cv::Mat R_stereo;
  cv::Rodrigues(r_stereo, R_stereo);
  std::cout << "Optimized - R_stereo = \n" << R_stereo << std::endl;
  std::cout << "Optimized - t_stereo = \n" << t_stereo << std::endl;

  // 5. Evaluation
  // w/o Stereo BA
  evaluateStereoCalibration(left_camera, right_camera,
                            K_left, D_left, K_right, D_right, R_stereo_init, t_stereo_init,
                            commonCorners_left, commonCorners_right);
  // w/ Stereo BA
  evaluateStereoCalibration(left_camera, right_camera,
                            K_left_optim, D_left_optim, K_right_optim, D_right_optim, R_stereo, t_stereo,
                            commonCorners_left, commonCorners_right);

  return 0;
}
