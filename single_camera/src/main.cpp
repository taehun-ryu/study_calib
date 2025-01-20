#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <ceres/ceres.h> // non-linear optimization libraries
// Custom headers
#include "Camera.hpp"
#include "CharucoBoard.hpp"

//TODO: Eigen이랑 혼합해서 사용하는 방법 찾아보기

namespace RefinementConfig {
  const cv::Size subPixWindowSize(5, 5);
  const cv::Size zeroZone(-1, -1);
  const cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);
}

void detectAndRefineCorners(SingleCamera& camera, CharucoBoard_5_5& board, std::vector<std::vector<cv::Point2f>>& allCorners, std::vector<std::vector<int>>& allIds)
{
  // Get the ChArUco board and dictionary
  cv::Ptr<cv::aruco::CharucoBoard> charucoBoard = board.getBoard();
  cv::Ptr<cv::aruco::Dictionary> dictionary = board.getDictionary();
  // Initialize vectors to store corners and ids
  // Iterate through the images
  for (size_t i = 0; i < camera.size(); ++i)
  {
    // Load image
    cv::Mat image = camera.getImage(i);
    if (image.empty())
    {
      std::cerr << "Failed to load image at index " << i << std::endl;
      continue;
    }
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect ArUco markers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds);
    if (markerIds.empty())
    {
      std::cout << "No markers detected in image " << i << std::endl;
      continue;
    }
    // Detect Charuco corners
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, gray, charucoBoard, charucoCorners, charucoIds);
    if (charucoIds.empty())
    {
      std::cout << "No ChArUco corners detected in image " << i << std::endl;
      continue;
    }
    // Refine ChArUco corners using subpixel accuracy
    cv::cornerSubPix(gray, charucoCorners, RefinementConfig::subPixWindowSize, RefinementConfig::zeroZone, RefinementConfig::termCriteria);
    // insert the detected corners and ids
    allCorners.push_back(charucoCorners);
    allIds.push_back(charucoIds);
  }
}

cv::Mat findHomographyForEachImage(std::vector<cv::Point2f>& objPoints, std::vector<cv::Point2f>& imgPoints)
{
  // Check if the number of object points and image points
  CV_Assert(objPoints.size() == imgPoints.size());

  // Build the matrix A in A * h = 0
  cv::Mat A(2 * objPoints.size(), 9, CV_64F);
  for (size_t i = 0; i < objPoints.size(); ++i)
  {
    const cv::Point2f& objPoint = objPoints[i];
    const cv::Point2f& imgPoint = imgPoints[i];
    // First row
    A.at<double>(2 * i, 0) = objPoint.x;
    A.at<double>(2 * i, 1) = objPoint.y;
    A.at<double>(2 * i, 2) = 1.0;
    A.at<double>(2 * i, 3) = 0.0;
    A.at<double>(2 * i, 4) = 0.0;
    A.at<double>(2 * i, 5) = 0.0;
    A.at<double>(2 * i, 6) = -objPoint.x * imgPoint.x;
    A.at<double>(2 * i, 7) = -objPoint.y * imgPoint.x;
    A.at<double>(2 * i, 8) = -imgPoint.x;
    // Second row
    A.at<double>(2 * i + 1, 0) = 0.0;
    A.at<double>(2 * i + 1, 1) = 0.0;
    A.at<double>(2 * i + 1, 2) = 0.0;
    A.at<double>(2 * i + 1, 3) = objPoint.x;
    A.at<double>(2 * i + 1, 4) = objPoint.y;
    A.at<double>(2 * i + 1, 5) = 1.0;
    A.at<double>(2 * i + 1, 6) = -objPoint.x * imgPoint.y;
    A.at<double>(2 * i + 1, 7) = -objPoint.y * imgPoint.y;
    A.at<double>(2 * i + 1, 8) = -imgPoint.y;
  }
  // Solve the equation using SVD
  cv::Mat w, u, vt;
  cv::SVD::compute(A, w, u, vt);
  cv::Mat homography = vt.row(8).reshape(1, 3);
  // Normalize the homography matrix to make h33 = 1
  homography /= homography.at<double>(2, 2);

  return homography;
}

double computeProjectionError(const cv::Mat& image,
                              const std::vector<cv::Point3f>& objPoints,
                              const std::vector<cv::Point2f>& imgPoints,
                              const cv::Mat& H)
{
  cv::Mat visualization = image.clone();
  double totalError = 0.0;

  for (size_t i = 0; i < objPoints.size(); ++i) {
    // Convert 3D object point to homogeneous coordinates
    cv::Mat point3D = (cv::Mat_<double>(3, 1) << objPoints[i].x, objPoints[i].y, 1.0);
    cv::Mat projectedPoint = H * point3D;

    // Compute the projected 2D point
    double x = projectedPoint.at<double>(0) / projectedPoint.at<double>(2);
    double y = projectedPoint.at<double>(1) / projectedPoint.at<double>(2);

    // Compute the reprojection error
    double error = cv::norm(cv::Point2f(x, y) - imgPoints[i]);
    totalError += error;

    // Draw the projected point and the image point with a circle
    cv::circle(visualization, cv::Point2f(x, y), 5, cv::Scalar(0, 255, 0), 2); // Projected point
    cv::circle(visualization, imgPoints[i], 5, cv::Scalar(255, 0, 0), 2); // Original point

    // Add the coordinates of the projected point as text
    std::string text = "(" + std::to_string(static_cast<int>(x)) + ", " + std::to_string(static_cast<int>(y)) + ")";
    cv::putText(visualization, text, cv::Point2f(x + 10, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Optionally add the image point coordinates as well
    text = "(" + std::to_string(static_cast<int>(imgPoints[i].x)) + ", " + std::to_string(static_cast<int>(imgPoints[i].y)) + ")";
    cv::putText(visualization, text, cv::Point2f(imgPoints[i].x + 10, imgPoints[i].y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  }

  // Display the visualization
  // cv::imshow("3D projection visualization", visualization);
  // cv::waitKey(0);

  return totalError / objPoints.size();
}


void normalizePoints(const std::vector<cv::Point2f>& points,
                     std::vector<cv::Point2f>& normalizedPoints,
                     cv::Mat& normalizationMat)
{
  // Calculate the normalization matrix (e.g., centroid and scale)
  cv::Point2f centroid(0, 0);
  for (const auto& point : points) {
    centroid += point;
  }
  centroid *= (1.0f / points.size());

  double avgDistance = 0.0;
  for (const auto& point : points) {
    avgDistance += cv::norm(point - centroid);
  }
  avgDistance /= points.size();

  double scale = std::sqrt(2.0) / avgDistance;

  normalizationMat = cv::Mat::eye(3, 3, CV_64F);
  normalizationMat.at<double>(0, 0) = scale;
  normalizationMat.at<double>(1, 1) = scale;
  normalizationMat.at<double>(0, 2) = -scale * centroid.x;
  normalizationMat.at<double>(1, 2) = -scale * centroid.y;

  normalizedPoints.clear();
  for (const auto& point : points) {
    cv::Mat pt = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
    cv::Mat normPt = normalizationMat * pt;
    normalizedPoints.emplace_back(normPt.at<double>(0) / normPt.at<double>(2),
                                  normPt.at<double>(1) / normPt.at<double>(2));
  }
}

cv::Mat calculateNormalizationMat(std::vector<cv::Point2f>& points)
{
  // Calculate the centroid of the points
  cv::Point2f centroid(0.0, 0.0);
  for (const cv::Point2f& point : points) {
    centroid += point;
  }
  centroid.x /= static_cast<float>(points.size());
  centroid.y /= static_cast<float>(points.size());

  // Calculate the average distance from the centroid
  double avgDistance = 0.0;
  for (const cv::Point2f& point : points) {
    avgDistance += cv::norm(point - centroid);
  }
  avgDistance /= static_cast<double>(points.size());

  // Calculate the scaling factor
  double scale = std::sqrt(2.0) / avgDistance;

  // Construct the normalization matrix
  cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
  T.at<double>(0, 0) = scale;
  T.at<double>(1, 1) = scale;
  T.at<double>(0, 2) = -scale * centroid.x;
  T.at<double>(1, 2) = -scale * centroid.y;

  return T;
}

// Homography Residual Function
struct HomographyResidual
{
  HomographyResidual(const cv::Point2f& objPoint, const cv::Point2f& imgPoint)
      : objPoint_(objPoint), imgPoint_(imgPoint) {}

  template <typename T>
  bool operator()(const T* const H, T* residual) const {
    // Projected points
    T x = T(objPoint_.x);
    T y = T(objPoint_.y);
    T w = H[6] * x + H[7] * y + H[8]; // w = h31*x + h32*y + h33

    T projected_x = (H[0] * x + H[1] * y + H[2]) / w; // x' = (h11*x + h12*y + h13) / w
    T projected_y = (H[3] * x + H[4] * y + H[5]) / w; // y' = (h21*x + h22*y + h23) / w

    // Residual = observed point - projected point
    residual[0] = T(imgPoint_.x) - projected_x;
    residual[1] = T(imgPoint_.y) - projected_y;

    return true;
  }

 private:
  const cv::Point2f objPoint_;
  const cv::Point2f imgPoint_;
};

cv::Mat optimizeHomography(const cv::Mat& H_init,
                           const std::vector<cv::Point2f>& objPoints,
                           const std::vector<cv::Point2f>& imgPoints)
{
  double H[9];
  for (int i = 0; i < 9; ++i) {
    H[i] = H_init.at<double>(i / 3, i % 3);
  }
  // Problem 세팅
  ceres::Problem problem;
  for (size_t i = 0; i < objPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HomographyResidual, 2, 9>( // residual dimension: 2, homography's dimension: 9
            new HomographyResidual(objPoints[i], imgPoints[i])),
        nullptr,  // 손실 함수
        H);       // 최적화 변수
  }

  // Solver 옵션 설정
  ceres::Solver::Options options;
  // Levenberg-Marquardt
  options.linear_solver_type = ceres::CGNR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.initial_trust_region_radius = 1e4;
  options.min_lm_diagonal = 1e-6;
  options.max_lm_diagonal = 1e32;

  // print
  options.minimizer_progress_to_stdout = false;

  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  //std::cout << summary.FullReport() << "\n";

  cv::Mat H_optimized(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    H_optimized.at<double>(i / 3, i % 3) = H[i];
  }

  return H_optimized;
}



cv::Mat estimateInitialIntrinsic(const std::vector<cv::Mat>& homographies) {
  cv::Mat A = cv::Mat::zeros(2 * homographies.size(), 6, CV_64F);

  for (size_t i = 0; i < homographies.size(); ++i) {
    cv::Mat H = homographies[i];

    double h11 = H.at<double>(0, 0);
    double h12 = H.at<double>(0, 1);
    double h21 = H.at<double>(1, 0);
    double h22 = H.at<double>(1, 1);
    double h31 = H.at<double>(2, 0);
    double h32 = H.at<double>(2, 1);

    // Populate matrix A
    A.at<double>(2 * i, 0) = h11 * h12;
    A.at<double>(2 * i, 1) = h11 * h22 + h21 * h12;
    A.at<double>(2 * i, 2) = h21 * h22;
    A.at<double>(2 * i, 3) = h31 * h12;
    A.at<double>(2 * i, 4) = h31 * h22 + h32 * h12;
    A.at<double>(2 * i, 5) = h32 * h22;

    A.at<double>(2 * i + 1, 0) = h11 * h11 - h12 * h12;
    A.at<double>(2 * i + 1, 1) = 2 * (h11 * h21 - h12 * h22);
    A.at<double>(2 * i + 1, 2) = h21 * h21 - h22 * h22;
    A.at<double>(2 * i + 1, 3) = h31 * h11 - h32 * h12;
    A.at<double>(2 * i + 1, 4) = 2 * (h31 * h21 - h32 * h22);
    A.at<double>(2 * i + 1, 5) = h31 * h31 - h32 * h32;
  }

  // Solve A * b = 0
  cv::Mat w, u, vt;
  cv::SVD::compute(A, w, u, vt);
  cv::Mat b = vt.row(vt.rows - 1).t();
  std::cout << "b: " << b << std::endl;

  // Extract intrinsic parameters
  double omega = b.at<double>(0)*b.at<double>(2)*b.at<double>(5) - b.at<double>(1)*b.at<double>(1)*b.at<double>(5)
                 - b.at<double>(0)*b.at<double>(4)*b.at<double>(4) + 2 * b.at<double>(1)*b.at<double>(3)*b.at<double>(4)
                 - b.at<double>(2)*b.at<double>(3)*b.at<double>(3);
  double d = b.at<double>(0)*b.at<double>(2) - b.at<double>(1)*b.at<double>(1);

  std::cout << "Omega: " << omega << std::endl;
  std::cout << "d: " << d << std::endl;

  // Check for invalid values
  // if (d == 0 || b.at<double>(0) == 0) {
  //   std::cerr << "Error: Division by zero detected. Returning identity matrix." << std::endl;
  //   return cv::Mat::eye(3, 3, CV_64F); // Return identity matrix as fallback
  // }
  // if (omega < 0) {
  //   std::cerr << "Error: Omega is negative. Cannot compute square root." << std::endl;
  //   return cv::Mat::eye(3, 3, CV_64F);
  // }

  // Compute intrinsic parameters safely
  double alpha = std::sqrt(omega / (d * b.at<double>(0)));
  double beta = std::sqrt((omega / (d * d)) * b.at<double>(0));
  double gamma = std::sqrt(std::abs(omega / (d * d * b.at<double>(0)))) * b.at<double>(1); // abs to prevent nan
  double uc = (b.at<double>(1)*b.at<double>(4) - b.at<double>(2)*b.at<double>(3)) / d;
  double vc = (b.at<double>(1)*b.at<double>(3) - b.at<double>(0)*b.at<double>(4)) / d;

  cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << alpha, gamma, uc,
                                                 0, beta, vc,
                                                 0, 0, 1);

  return intrinsic;
}

// Helper function to calculate the V matrix from homographies
cv::Mat constructV(const cv::Mat& H, int i, int j) {
  double hij1 = H.at<double>(0, i) * H.at<double>(0, j);
  double hij2 = H.at<double>(1, i) * H.at<double>(1, j);
  double hij3 = H.at<double>(2, i) * H.at<double>(2, j);

  cv::Mat v = (cv::Mat_<double>(1, 6) << hij1,
                                         H.at<double>(0, i) * H.at<double>(1, j) + H.at<double>(1, i) * H.at<double>(0, j),
                                         hij2,
                                         H.at<double>(2, i) * H.at<double>(0, j) + H.at<double>(0, i) * H.at<double>(2, j),
                                         H.at<double>(2, i) * H.at<double>(1, j) + H.at<double>(1, i) * H.at<double>(2, j),
                                         hij3);
  return v;
}

cv::Mat estimateInitialIntrinsicLLT(const std::vector<cv::Mat>& homographies) {
  cv::Mat V;  // Will hold all the V matrices
  for (const auto& H : homographies) {
    // Construct V matrix rows for each homography
    cv::Mat v12 = constructV(H, 0, 1);  // v_12
    cv::Mat v11 = constructV(H, 0, 0);  // v_11
    cv::Mat v22 = constructV(H, 1, 1);  // v_22
    V.push_back(v12);
    V.push_back(v11 - v22);  // Use the linear constraints for intrinsic
  }

  // Solve V * b = 0 using SVD
  cv::Mat b;
  cv::SVD::solveZ(V, b);

  // Intrinsic matrix components from b
  double B11 = b.at<double>(0, 0);
  double B12 = b.at<double>(1, 0);
  double B22 = b.at<double>(2, 0);
  double B13 = b.at<double>(3, 0);
  double B23 = b.at<double>(4, 0);
  double B33 = b.at<double>(5, 0);

  // Camera intrinsic matrix K from B matrix
  Eigen::Matrix3d B;
  B << B11, B12, B13,
       B12, B22, B23,
       B13, B23, B33;

  // Cholesky decomposition: K^-T * K^-1 = B
  Eigen::LLT<Eigen::Matrix3d> lltOfB(B);
  Eigen::Matrix3d K_inv_T = lltOfB.matrixL();  // Lower triangular matrix

  // Invert and transpose to get K
  Eigen::Matrix3d K_eigen = K_inv_T.transpose().inverse();

  // Convert Eigen matrix to OpenCV matrix
  cv::Mat K(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      K.at<double>(i, j) = K_eigen(i, j);

  // Normalize K so K(2,2) = 1
  K /= K.at<double>(2, 2);

  return K;
}

void initializeExtrinsic(const std::vector<cv::Mat>& homographies, const cv::Mat& K,
                         std::vector<cv::Mat>& rotationMatrices,
                         std::vector<cv::Mat>& translationVectors)
{
  for (size_t i = 0; i < homographies.size(); i++)
  {
    cv::Mat H = homographies[i];

    // Decompose homography to [R|t]
    cv::Mat R(3, 3, CV_64F);
    cv::Mat t(3, 1, CV_64F);

    // Extract columns of H
    cv::Mat h1 = H.col(0);
    cv::Mat h2 = H.col(1);
    cv::Mat h3 = H.col(2);

    // Compute scaling factor lambda
    double lambda = 1.0 / cv::norm(K.inv() * h1);

    // Compute rotation and translation
    cv::Mat r1 = lambda * K.inv() * h1;
    cv::Mat r2 = lambda * K.inv() * h2;
    t = lambda * K.inv() * h3;

    // Ensure R is a valid rotation matrix using SVD
    cv::Mat r3 = r1.cross(r2);
    R = (cv::Mat_<double>(3, 3) << r1.at<double>(0), r2.at<double>(0), r3.at<double>(0),
                                   r1.at<double>(1), r2.at<double>(1), r3.at<double>(1),
                                   r1.at<double>(2), r2.at<double>(2), r3.at<double>(2));

    cv::SVD svd(R, cv::SVD::FULL_UV);
    R = svd.u * svd.vt; // Force orthogonality

    // Store the results
    rotationMatrices.push_back(R);
    translationVectors.push_back(t);
  }
}

int main() {
  // Initialize SingleCamera and CharucoBoard
  SingleCamera camera("/home/user/calib_data/1204_stereo/Cam_001/");
  CharucoBoard_5_5 board(0, 11);

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
    double error = computeProjectionError(camera.getImage(i), allObjPoints3D[i], allCornersImg[i], H);
    homographies.push_back(H);
  }
  // 3. Estimate initial intrinsic matrix
  cv::Mat K = estimateInitialIntrinsicLLT(homographies);
  std::cout << "K: " << K << std::endl;

  // 4. Estimate initial extrinsic matrix
  std::vector<cv::Mat> rotationMatrices;
  std::vector<cv::Mat> translationVectors;
  initializeExtrinsic(homographies, K, rotationMatrices, translationVectors);

  // 5. Estimate radial lens distortion

  return 0;
}
