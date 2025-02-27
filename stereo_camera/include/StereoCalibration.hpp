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
  options.minimizer_progress_to_stdout = false;

  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  convertVecArray2VecCVMat(all_rvecs, rvecs);
  convertVecArray2VecCVMat(all_tvecs, tvecs);
#if SKEW_COEFFICIENT
  K_optim = (cv::Mat_<double>(3, 3) << K[0], K[4], K[2],
                                        0.0, K[1], K[3],
                                        0.0, 0.0, 1.0);
#else
  K_optim = (cv::Mat_<double>(3, 3) << K[0], 0.0, K[2],
                                        0.0, K[1], K[3],
                                        0.0, 0.0, 1.0);
#endif
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
    cv::circle(vis_left, pt, 5, cv::Scalar(0,0,255), -1);
  for (const auto& pt : unique_right)
    cv::circle(vis_right, pt, 5, cv::Scalar(0,0,255), -1);
  for (const auto& pt : corners_common_left)
    cv::circle(vis_left, pt, 5, cv::Scalar(0,255,0), -1);
  for (const auto& pt : corners_common_right)
    cv::circle(vis_right, pt, 5, cv::Scalar(0,255,0), -1);

  cv::Mat concat;
  cv::hconcat(vis_left, vis_right, concat);

  cv::imshow("Common Corners", concat);
  cv::waitKey(0);
}

cv::Vec4d rotationMatrixToQuaternion(const cv::Mat& R) {
  double m00 = R.at<double>(0,0), m01 = R.at<double>(0,1), m02 = R.at<double>(0,2);
  double m10 = R.at<double>(1,0), m11 = R.at<double>(1,1), m12 = R.at<double>(1,2);
  double m20 = R.at<double>(2,0), m21 = R.at<double>(2,1), m22 = R.at<double>(2,2);
  double trace = m00 + m11 + m22;
  double w, x, y, z;
  if (trace > 0.0) {
    double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else {
    if ((m00 > m11) && (m00 > m22)) {
      double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
      w = (m21 - m12) / s;
      x = 0.25 * s;
      y = (m01 + m10) / s;
      z = (m02 + m20) / s;
    } else if (m11 > m22) {
      double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
      w = (m02 - m20) / s;
      x = (m01 + m10) / s;
      y = 0.25 * s;
      z = (m12 + m21) / s;
    } else {
      double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
      w = (m10 - m01) / s;
      x = (m02 + m20) / s;
      y = (m12 + m21) / s;
      z = 0.25 * s;
    }
  }
  return cv::Vec4d(w, x, y, z);
}

cv::Mat quaternionToRotationMatrix(const cv::Vec4d& q) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double xx = x * x, yy = y * y, zz = z * z;
  double xy = x * y, xz = x * z, yz = y * z;
  double wx = w * x, wy = w * y, wz = w * z;
  cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
  R.at<double>(0,0) = 1.0 - 2.0*(yy + zz);
  R.at<double>(0,1) = 2.0*(xy - wz);
  R.at<double>(0,2) = 2.0*(xz + wy);
  R.at<double>(1,0) = 2.0*(xy + wz);
  R.at<double>(1,1) = 1.0 - 2.0*(xx + zz);
  R.at<double>(1,2) = 2.0*(yz - wx);
  R.at<double>(2,0) = 2.0*(xz - wy);
  R.at<double>(2,1) = 2.0*(yz + wx);
  R.at<double>(2,2) = 1.0 - 2.0*(xx + yy);
  return R;
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
  for (int i = 0; i < 1; i++)
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

// -----------------------------
// Reprojection Error Cost Functor
// -----------------------------
struct StereoReprojectionResidual
{
  StereoReprojectionResidual(double common_obj_x, double common_obj_y,
                            double left_img_x, double left_img_y,
                            double right_img_x , double right_img_y) :
    common_obj_x_(common_obj_x), common_obj_y_(common_obj_y), left_img_x_(left_img_x), left_img_y_(left_img_y), right_img_x_(right_img_x), right_img_y_(right_img_y) {}

  template <typename T>
  bool operator()(const T* const K_left, const T* const d_left,
                  const T* const K_right, const T* const d_right,
                  const T* const rvec_left, const T* const tvec_left,
                  const T* const rvec_right, const T* const tvec_right,
                  const T* const rvec_stereo, const T* tvec_stereo,
                  T* residual) const
  {
    T object_pt[3] = {T(common_obj_x_), T(common_obj_y_), T(0.0)};

    // 1) World -> Left camera coordinate -> Right camera coordinate
    auto right_pt_from_left = transformWorldLeftRight(object_pt, rvec_stereo, tvec_stereo, rvec_left, tvec_left);
    // 2) Right camera coordinate -> Right camera's pixel coordinate
    auto right_proj_from_left = projetCamera2Pixel(right_pt_from_left.data(), K_right, d_right);

    // 1) World -> Right camera coordinate -> Left camera coordinate
    auto left_pt_from_left = transformWorldRightLeft(object_pt, rvec_stereo, tvec_stereo, rvec_right, tvec_right);
    // 2) Left camera coordinate -> Left camera's pixel coordinate
    auto left_proj_from_right = projetCamera2Pixel(left_pt_from_left.data(), K_left, d_left);

    residual[0] = right_proj_from_left[0] - T(right_img_x_);
    residual[1] = right_proj_from_left[1] - T(right_img_y_);
    residual[2] = left_proj_from_right[0] - T(left_img_x_);
    residual[3] = left_proj_from_right[1] - T(left_img_y_);

    return true;
  }

  static ceres::CostFunction* Create(double common_obj_x, double common_obj_y,
                                     double left_img_x, double left_img_y,
                                     double right_img_x , double right_img_y)
  {
#if SKEW_COEFFICIENT
    return (new ceres::AutoDiffCostFunction<StereoReprojectionResidual, 4, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3>(
        new StereoReprojectionResidual(common_obj_x, common_obj_y, left_img_x, left_img_y, right_img_x, right_img_y)));
#else
    return (new ceres::AutoDiffCostFunction<StereoReprojectionResidual, 4, 4, 5, 4, 5, 3, 3, 3, 3, 3, 3>(
        new StereoReprojectionResidual(common_obj_x, common_obj_y, left_img_x, left_img_y, right_img_x, right_img_y)));
#endif
  }

 private:
  double common_obj_x_, common_obj_y_;
  double left_img_x_, left_img_y_;
  double right_img_x_, right_img_y_;

  template <typename T>
  std::array<T, 2> projetCamera2Pixel(const T *points3D, const T *K, const T *d)  const;
  template <typename T>
  std::array<T, 3> transformWorldLeftRight(const T* srcPoints, const T* rvec_stereo, const T* tvec_stereo, const T *rvec, const T *tvec) const;
  template <typename T>
  std::array<T, 3> transformWorldRightLeft(const T* srcPoints, const T* rvec_stereo, const T* tvec_stereo, const T *rvec, const T *tvec) const;
};

template<typename T>
std::array<T, 2> StereoReprojectionResidual::projetCamera2Pixel(const T* points3D, const T* K, const T* d) const
{
  T p_c[3] = {T(points3D[0]), T(points3D[1]), T(points3D[2])};

  T x = p_c[0] / p_c[2];
  T y = p_c[1] / p_c[2];

  // d = [k1, k2, p1, p2, k3]
  T r2 = x * x + y * y;
  T radial_distortion = T(1.0) + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
  T x_distorted = x * radial_distortion + T(2.0) * d[2] * x * y + d[3] * (r2 + T(2.0) * x * x);
  T y_distorted = y * radial_distortion + d[2] * (r2 + T(2.0) * y * y) + T(2.0) * d[3] * x * y;

#if SKEW_COEFFICIENT
  // K = [fx, fy, cx, cy, s]
  T predicted_x = K[0] * x_distorted + K[4] * y_distorted + K[2];
#else
  T predicted_x = K[0] * x_distorted + K[2];
#endif
  T predicted_y = K[1] * y_distorted + K[3];

  return {predicted_x, predicted_y};
}

template<typename T>
std::array<T, 3> StereoReprojectionResidual::transformWorldLeftRight(const T* srcPoints, const T* rvec_stereo, const T* tvec_stereo, const T *rvec, const T *tvec) const
{
  // World -> Left camera coordinates
  T p_w[3] = {T(srcPoints[0]), T(srcPoints[1]), T(0.0)};
  T p_c[3];
  ceres::AngleAxisRotatePoint(rvec, p_w, p_c);
  p_c[0] += tvec[0];
  p_c[1] += tvec[1];
  p_c[2] += tvec[2];

  // Left camera coordinates -> Right camera coordinates
  std::array<T, 3> dst;
  ceres::AngleAxisRotatePoint(rvec_stereo, p_c, dst.data());
  dst[0] += tvec_stereo[0];
  dst[1] += tvec_stereo[1];
  dst[2] += tvec_stereo[2];

  return dst;
}

template<typename T>
std::array<T, 3> StereoReprojectionResidual::transformWorldRightLeft(const T* srcPoints, const T* rvec_stereo, const T* tvec_stereo, const T *rvec, const T *tvec) const
{
  // World -> Right camera coordinates
  T p_w[3] = {T(srcPoints[0]), T(srcPoints[1]), T(0.0)};
  T p_c[3];
  ceres::AngleAxisRotatePoint(rvec, p_w, p_c);
  p_c[0] += tvec[0];
  p_c[1] += tvec[1];
  p_c[2] += tvec[2];

  // 1) Left -> Right: rvec_stereo, tvec_stereo
  //    Right -> Left: -rvec_stereo,  R_lr^T * (-tvec_stereo)
  T inverse_rvec[3];
  inverse_rvec[0] = -rvec_stereo[0];
  inverse_rvec[1] = -rvec_stereo[1];
  inverse_rvec[2] = -rvec_stereo[2];

  T p_c_minus_t[3];
  p_c_minus_t[0] = p_c[0] - tvec_stereo[0];
  p_c_minus_t[1] = p_c[1] - tvec_stereo[1];
  p_c_minus_t[2] = p_c[2] - tvec_stereo[2];

  // R_lr^T( p_c - t_lr )
  std::array<T, 3> dst;
  ceres::AngleAxisRotatePoint(inverse_rvec, p_c_minus_t, dst.data());

  return dst;
}

// -----------------------------
// Epipolar Constraint Cost Functor
// -----------------------------
struct StereoEpipolarResidual {
  StereoEpipolarResidual(double left_img_x, double left_img_y,
                           double right_img_x, double right_img_y)
    : left_img_x_(left_img_x), left_img_y_(left_img_y),
      right_img_x_(right_img_x), right_img_y_(right_img_y) {}

  // Parameter blocks:
  // K_left: 5 parameters ([fx, fy, cx, cy, skew])
  // K_right: 5 parameters ([fx, fy, cx, cy, skew])
  // rvec: 3 parameters (global stereo rotation in angle-axis)
  // tvec: 3 parameters (global stereo translation)
  template <typename T>
  bool operator()(const T* const K_left, const T* const K_right,
                  const T* const rvec, const T* const tvec,
                  T* residual) const
  {
    // Convert angle-axis rvec to rotation matrix R (3x3)
    T R[9];
    ceres::AngleAxisToRotationMatrix(rvec, R);

    // Construct skew-symmetric matrix [t]_x from tvec.
    T tx[9];
    tx[0] = T(0);      tx[1] = -tvec[2];  tx[2] = tvec[1];
    tx[3] = tvec[2];   tx[4] = T(0);      tx[5] = -tvec[0];
    tx[6] = -tvec[1];  tx[7] = tvec[0];   tx[8] = T(0);

    // Essential matrix: E = [t]_x * R.
    T E[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        E[3 * i + j] = T(0);
        for (int k = 0; k < 3; ++k) {
          E[3 * i + j] += tx[3 * i + k] * R[3 * k + j];
        }
      }
    }

    // Compute inverse of K_left and (K_right^{-1})^T.
    T K_left_inv[9];
    ComputeIntrinsicInverse(K_left, K_left_inv);
    T K_right_invT[9];
    ComputeIntrinsicRightInvT(K_right, K_right_invT);

    // Fundamental matrix: F = K_right^{-T} * E * K_left^{-1}
    T temp[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        temp[3 * i + j] = T(0);
        for (int k = 0; k < 3; ++k) {
          temp[3 * i + j] += E[3 * i + k] * K_left_inv[3 * k + j];
        }
      }
    }
    T F[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        F[3 * i + j] = T(0);
        for (int k = 0; k < 3; ++k) {
          F[3 * i + j] += K_right_invT[3 * i + k] * temp[3 * k + j];
        }
      }
    }

    // Define homogeneous image coordinates.
    T x_left[3]  = { T(left_img_x_), T(left_img_y_), T(1.0) };
    T x_right[3] = { T(right_img_x_), T(right_img_y_), T(1.0) };

    // Compute F * x_left.
    T Fx[3];
    for (int i = 0; i < 3; ++i) {
      Fx[i] = T(0);
      for (int j = 0; j < 3; ++j) {
        Fx[i] += F[3 * i + j] * x_left[j];
      }
    }
    // Compute F^T * x_right.
    T FTx[3];
    for (int i = 0; i < 3; ++i) {
      FTx[i] = T(0);
      for (int j = 0; j < 3; ++j) {
        // F^T(i,j) = F(3*j + i)
        FTx[i] += F[3 * j + i] * x_right[j];
      }
    }

    // Compute the epipolar constraint value: x_right^T * F * x_left.
    T xTFx = T(0);
    for (int i = 0; i < 3; ++i) {
      xTFx += x_right[i] * Fx[i];
    }

    // Compute denominator for Sampson error:
    // denom = sqrt((Fx[0])^2 + (Fx[1])^2 + (FTx[0])^2 + (FTx[1])^2 + epsilon)
    T epsilon = T(1e-8);
    T denom = ceres::sqrt(Fx[0] * Fx[0] + Fx[1] * Fx[1] +
                           FTx[0] * FTx[0] + FTx[1] * FTx[1] + epsilon);
    // Compute the final residual.
    if (denom < T(1e-12)) {
      residual[0] = T(0);
    } else {
      residual[0] = xTFx / denom;
    }
    return true;
  }

  // Factory method.
  static ceres::CostFunction* Create(double left_img_x, double left_img_y,
                                     double right_img_x, double right_img_y)
  {
#if SKEW_COEFFICIENT
    return (new ceres::AutoDiffCostFunction<StereoEpipolarResidual, 1, 5, 5, 3, 3>(
        new StereoEpipolarResidual(left_img_x, left_img_y, right_img_x, right_img_y)));
#else
    return (new ceres::AutoDiffCostFunction<StereoEpipolarResidual, 1, 4, 4, 3, 3>(
        new StereoEpipolarResidual(left_img_x, left_img_y, right_img_x, right_img_y)));
#endif
  }

 private:
  const double left_img_x_, left_img_y_;
  const double right_img_x_, right_img_y_;

  template <typename T>
  static void ComputeIntrinsicInverse(const T* const K, T* K_inv);

  template <typename T>
  static void ComputeIntrinsicRightInvT(const T* const K, T* K_right_invT);
};

// Given intrinsic K (5 parameters: [fx, fy, cx, cy, skew]),
// compute K_inv (3x3, row-major order).
template <typename T>
void StereoEpipolarResidual::ComputeIntrinsicInverse(const T* const K, T* K_inv) {
  // K = [fx, fy, cx, cy, skew]
  T fx = K[0];
  T fy = K[1];
  T cx = K[2];
  T cy = K[3];
#if SKEW_COEFFICIENT
  T skew = K[4];
#else
  T skew = T(0.0);
#endif
  // Prevent division by zero.
  if (fx < T(1e-8)) fx = T(1e-8);
  if (fy < T(1e-8)) fy = T(1e-8);
  K_inv[0] = T(1) / fx;
  K_inv[1] = -skew / (fx * fy);
  K_inv[2] = (skew * cy - cx * fy) / (fx * fy);
  K_inv[3] = T(0);
  K_inv[4] = T(1) / fy;
  K_inv[5] = -cy / fy;
  K_inv[6] = T(0);
  K_inv[7] = T(0);
  K_inv[8] = T(1);
}

// Compute K_right_invT = (K_right^{-1})^T.
template <typename T>
void StereoEpipolarResidual::ComputeIntrinsicRightInvT(const T* const K, T* K_right_invT) {
  T K_inv[9];
  ComputeIntrinsicInverse(K, K_inv);
  // Transpose K_inv.
  K_right_invT[0] = K_inv[0];
  K_right_invT[1] = K_inv[3];
  K_right_invT[2] = K_inv[6];
  K_right_invT[3] = K_inv[1];
  K_right_invT[4] = K_inv[4];
  K_right_invT[5] = K_inv[7];
  K_right_invT[6] = K_inv[2];
  K_right_invT[7] = K_inv[5];
  K_right_invT[8] = K_inv[8];
}