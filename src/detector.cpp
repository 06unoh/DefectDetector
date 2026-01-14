#include "detector.h"
#include <cmath>
#include <cstdio>

cv::Mat DefectDetector::to_gray(const cv::Mat& bgr) {
  cv::Mat gray;
  if (bgr.channels() == 3) {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bgr.clone();
  }
  return gray;
}

double DefectDetector::calc_circularity(const std::vector<cv::Point>& contour) {
  double area = cv::contourArea(contour);
  double peri = cv::arcLength(contour, true);
  if (peri <= 1e-9) {
    return 0.0;
  }
  return (4.0 * CV_PI * area) / (peri * peri);
}

cv::Mat DefectDetector::make_mask_field(const cv::Mat& bgr_roi) {
  CV_Assert(bgr_roi.empty()==false);

  cv::Mat gray = to_gray(bgr_roi);

  // 약한 블러(너무 세면 얇은 스크래치가 죽음)
  if (params_.blur_ksize >= 3 && (params_.blur_ksize % 2 == 1)) {
    cv::GaussianBlur(gray, gray, cv::Size(params_.blur_ksize, params_.blur_ksize), 0);  
  }

  int bgk = params_.morph_ksize * 4;   
  bgk = std::clamp(bgk, 21, 31);       
  if (bgk % 2 == 0) bgk += 1;

  cv::Mat bg_g, res_g, m_res;
  cv::GaussianBlur(gray, bg_g, cv::Size(bgk, bgk), 0);
  cv::absdiff(gray, bg_g, res_g);   // 그레이 img - 그레이 배경 = 그레이 마스크 후보


  double t_res = cv::threshold(res_g, m_res, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);   
  double t_res2 = std::max(5.0, t_res * 0.6);
  cv::threshold(res_g, m_res, t_res2, 255, cv::THRESH_BINARY);


  cv::Mat edges;  
  {
    cv::Mat tmp = gray.reshape(1, 1);
    cv::Mat tmp_sorted;
    cv::sort(tmp, tmp_sorted, cv::SORT_ASCENDING);
    int mid = tmp_sorted.cols / 2;
    double med = tmp_sorted.at<uchar>(0, mid); 

    double low = 0.66 * med;
    double high = 1.33 * med;

    low  = std::clamp(low,  5.0,  30.0);
    high = std::clamp(high, 20.0, 80.0);

    cv::Canny(gray, edges, low, high);  


    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)); 
    cv::dilate(edges, edges, k3, cv::Point(-1,-1), 1); 
  }

  cv::Mat m_edge = edges; 

  cv::Mat lab;
  cv::cvtColor(bgr_roi, lab, cv::COLOR_BGR2Lab);  
  std::vector<cv::Mat> ch;
  cv::split(lab, ch); 

  auto make_res_mask = [&](const cv::Mat& src8u)->cv::Mat {
    cv::Mat bg, res, m;
    cv::GaussianBlur(src8u, bg, cv::Size(bgk, bgk), 0);
    cv::absdiff(src8u, bg, res);

    double t = cv::threshold(res, m, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double t2 = std::max(5.0, t * 0.6); 
    cv::threshold(res, m, t2, 255, cv::THRESH_BINARY);
    return m;
  };

  cv::Mat m_color = make_res_mask(ch[1]) | make_res_mask(ch[2]); 

  cv::Mat bin = m_res | m_edge | m_color; 

  int k = params_.morph_ksize;
  if (k < 3) {
    k = 3;
  }
  if (k % 2 == 0) {
    k += 1;
  }

  cv::Mat k_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
  cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k_close);     
  cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k_close);

  // 미세 스크래치 제거 -> 보류 
  // cv::Mat k_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
  // cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k_open);    

  return bin;
}


std::vector<DefectCandidate> DefectDetector::run(const cv::Mat& bgr, cv::Mat& mask_out) {
  CV_Assert(bgr.empty()==false);

  cv::Rect roi = params_.roi;
  if (roi.width <= 0 || roi.height <= 0) {
    roi = cv::Rect(0, 0, bgr.cols, bgr.rows);
  }

  roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);

  cv::Mat bgr_roi = bgr(roi); 
  cv::Mat mask_roi = make_mask_field(bgr_roi);

  // contour
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;  
  cv::findContours(mask_roi, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<DefectCandidate> out;
  out.reserve(contours.size());  

  for (const auto& c : contours) {
    double area = cv::contourArea(c);
    if (area < params_.area_min || area > params_.area_max) continue;

    cv::Rect bb = cv::boundingRect(c);
    double ar = (bb.height > 0) ? (static_cast<double>(bb.width) / bb.height) : 0.0;  
    if (ar < params_.aspect_min || ar > params_.aspect_max) continue;

    double circ = calc_circularity(c);
    if (params_.circ_min > 0.0 && circ < params_.circ_min) continue;

    DefectCandidate cand;
    cand.bbox = cv::Rect(bb.x + roi.x, bb.y + roi.y, bb.width, bb.height);
    cand.area = area;
    cand.aspect_ratio = ar;
    cand.circularity = circ;
    out.push_back(cand);
  }

  // mask_out: full size로 확장
  mask_out = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);   
  mask_roi.copyTo(mask_out(roi)); 

  return out;
}
