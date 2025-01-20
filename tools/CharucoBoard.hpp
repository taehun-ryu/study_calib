#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

/*
 * This is the charuco board list in {3D Vision & Robotics Lab}@UNIST.
*/
class CharucoBoard_6_9
{
 public:
  CharucoBoard_6_9(int minId, int maxId);
  ~CharucoBoard_6_9();
  cv::Ptr<cv::aruco::CharucoBoard> getBoard() { return board_; }
  cv::Ptr<cv::aruco::Dictionary> getDictionary() { return dictionary_; }

 private:
  cv::Ptr<cv::aruco::CharucoBoard> board_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

CharucoBoard_6_9::CharucoBoard_6_9(int minId, int maxId)
{
  dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
  board_ = cv::aruco::CharucoBoard::create(6, 9, 83.f, 62.f, dictionary_); //mm
  int id = minId;
  for (size_t i = 0; i <= board_->ids.size(); i++)
  {
    if (id > maxId + 1)
    {
      std::cout << "Warning: maxId reached" << std::endl;
      break;
    }
    board_->ids[i] = id++;
  }
}

CharucoBoard_6_9::~CharucoBoard_6_9() {}

class CharucoBoard_5_5
{
 public:
  CharucoBoard_5_5(int minId, int maxId);
  ~CharucoBoard_5_5();
  cv::Ptr<cv::aruco::CharucoBoard> getBoard() { return board_; }
  cv::Ptr<cv::aruco::Dictionary> getDictionary() { return dictionary_; }

 private:
  cv::Ptr<cv::aruco::CharucoBoard> board_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

CharucoBoard_5_5::CharucoBoard_5_5(int minId, int maxId)
{
  dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
  board_ = cv::aruco::CharucoBoard::create(5, 5, 98.f, 73.f, dictionary_); //mm
  int id = minId;
  for (size_t i = 0; i <= board_->ids.size(); i++)
  {
    if (id > maxId + 1)
    {
      std::cout << "Warning: maxId reached" << std::endl;
      break;
    }
    board_->ids[i] = id++;
  }
}

CharucoBoard_5_5::~CharucoBoard_5_5()
{
}

void showCharucoBoard(const cv::Ptr<cv::aruco::CharucoBoard>& board, const std::string& windowName)
{
  cv::Mat boardImage;
  board->draw(cv::Size(600, 500), boardImage, 10, 1);
  cv::imshow(windowName, boardImage);
  cv::waitKey(0);
}
