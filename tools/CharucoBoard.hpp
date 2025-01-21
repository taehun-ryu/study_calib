#pragma once
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

/*
 * This is the charuco board config list in {3D Vision & Robotics Lab}@UNIST.
*/

// Configuration structure for Charuco board
struct CharucoConfig {
  int rows;                // Number of rows (chessboard squares)
  int cols;                // Number of columns (chessboard squares)
  float squareLength;      // Length of a chessboard square (e.g., mm)
  float markerLength;      // Length of a marker inside the square (e.g., mm)
  int minId;               // Minimum marker ID
  int maxId;               // Maximum marker ID
  cv::aruco::PREDEFINED_DICTIONARY_NAME dictionary; // Predefined dictionary
};

// Configuration 1: 6x9 board, id: 0~26
CharucoConfig BoardConfig6x9_1 = {
  6,   // rows
  9,   // cols
  0.083f, // squareLength (e.g., m)
  0.062f, // markerLength (e.g., m)
  0,   // minId
  26,  // maxId
  cv::aruco::DICT_6X6_100 // dictionary type
};

// Configuration 2: 6x9 board, id: 27~53
CharucoConfig BoardConfig6x9_2 = {
  6,   // rows
  9,   // cols
  0.083f, // squareLength (e.g., m)
  0.062f, // markerLength (e.g., m)
  27,   // minId
  53,  // maxId
  cv::aruco::DICT_6X6_100 // dictionary type
};

// Configuration 3: 5x5 board, id: 0~11
CharucoConfig BoardConfig5x5 = {
  5,   // rows
  5,  // cols
  0.098f, // squareLength
  0.073f,  // markerLength
  0,   // minId
  11,  // maxId
  cv::aruco::DICT_6X6_100 // dictionary type
};

// CharucoBoard class supporting multiple configurations
class CharucoBoard {
 public:
  CharucoBoard(const CharucoConfig& config);
  ~CharucoBoard();

  cv::Ptr<cv::aruco::CharucoBoard> getBoard() const { return board_; }
  cv::Ptr<cv::aruco::Dictionary> getDictionary() const { return dictionary_; }
  void showCharucoBoard(const std::string& windowName);

 private:
  cv::Ptr<cv::aruco::CharucoBoard> board_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

// Constructor implementation
CharucoBoard::CharucoBoard(const CharucoConfig& config) {
  // Load dictionary
  dictionary_ = cv::aruco::getPredefinedDictionary(config.dictionary);
  // Create Charuco board
  board_ = cv::aruco::CharucoBoard::create(
      config.cols, config.rows, config.squareLength, config.markerLength, dictionary_);
  // Assign marker IDs based on minId and maxId
  int id = config.minId;
  for (size_t i = 0; i < board_->ids.size(); i++) {
    if (id > config.maxId) {
      std::cout << "Warning: maxId reached for configuration" << std::endl;
      break;
    }
    board_->ids[i] = id++;
  }
}

CharucoBoard::~CharucoBoard() {}

// Function to display a Charuco board
void CharucoBoard::showCharucoBoard(const std::string& windowName) {
  cv::Mat boardImage;
  getBoard()->draw(cv::Size(600, 500), boardImage, 10, 1);
  cv::imshow(windowName, boardImage);
  cv::waitKey(0);
}
