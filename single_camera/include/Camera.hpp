#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

class SingleCamera
{
 public:
  SingleCamera(const std::string& image_dir);
  ~SingleCamera();

  void loadImages();
  void setCameraParameters(const cv::Mat& cameraMatrix_, const cv::Mat& distCoeffs_);
  cv::Mat getCameraMatrix() { return cameraMatrix_; }
  cv::Mat getDistCoeffs() { return distCoeffs_; }
  cv::Mat getImage(size_t index) { return cv::imread(imageList_[index]); }
  size_t size() { return imageList_.size(); }

 private:
  std::string imageDirectory_;
  std::vector<std::string> imageList_;
  cv::Mat cameraMatrix_;
  cv::Mat distCoeffs_;
};

SingleCamera::SingleCamera(const std::string& image_dir) : imageDirectory_(image_dir)
{
  for (const auto& entry : std::filesystem::directory_iterator(imageDirectory_))
  {
    if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
    {
      imageList_.push_back(entry.path().string());
    }
  }
  std::cout << "Loaded " << imageList_.size() << " images from " << imageDirectory_ << std::endl;
}

SingleCamera::~SingleCamera() {}

void SingleCamera::setCameraParameters(const cv::Mat& cameraMatrix_, const cv::Mat& distCoeffs_)
{
  this->cameraMatrix_ = cameraMatrix_.clone();
  this->distCoeffs_ = distCoeffs_.clone();
}
