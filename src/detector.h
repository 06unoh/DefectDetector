#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DefectCandidate {
  cv::Rect bbox;
  double area = 0.0;  
  double aspect_ratio = 0.0;   
  double circularity = 0.0;    
};

struct DetectorParams {
  cv::Rect roi = cv::Rect();      
  std::string thresh = "otsu";    
  int blur_ksize = 3;          

  std::string morph = "both";     
  int morph_ksize = 9;          

  double area_min = 5.0;       
  double area_max = 1e9;     

  double aspect_min = 0.0;      
  double aspect_max = 1e9;

  double circ_min = 0.0;    // 스크래치 때문에 기본 0 권장
};

class DefectDetector {
public:
  explicit DefectDetector(const DetectorParams& p) : params_(p) {}

  std::vector<DefectCandidate> run(const cv::Mat& bgr, cv::Mat& mask_out);

private:
  DetectorParams params_;
  cv::Mat to_gray(const cv::Mat& bgr);
  double calc_circularity(const std::vector<cv::Point>& contour);   
  cv::Mat make_mask_field(const cv::Mat& bgr_roi);
};
