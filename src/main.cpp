#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm> 
#include <cstdio>

#include "detector.h"

namespace fs = std::filesystem;

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::string ts_ms_string() {
  using namespace std::chrono;
  auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  return std::to_string(ms);
}

static bool rect_equal(const cv::Rect& a, const cv::Rect& b) {
  return a.x==b.x && a.y==b.y && a.width==b.width && a.height==b.height;
}

static cv::Rect make_center_roi(int w, int h, double wr, double hr) {
  wr = std::max(0.01, std::min(1.0, wr));
  hr = std::max(0.01, std::min(1.0, hr));

  int rw = (int)std::round(w * wr);
  int rh = (int)std::round(h * hr);
  rw = std::max(1, std::min(rw, w));
  rh = std::max(1, std::min(rh, h));

  int rx = (w - rw) / 2;
  int ry = (h - rh) / 2;
  return cv::Rect(rx, ry, rw, rh);
}

static void save_snapshot_set(const std::string& save_dir, const std::string& prefix, const cv::Mat& raw, const cv::Mat& vis_ui, const cv::Mat& mask_full) {
  fs::create_directories(save_dir);
  std::string base = save_dir + "/" + prefix;

  bool success = cv::imwrite(base + "_raw.jpg", raw) && cv::imwrite(base + "_vis.jpg", vis_ui) && cv::imwrite(base + "_mask.png", mask_full);

  if (!success) {
      std::cerr << "save error " << base << "\n";
  } else {
      std::cout << "Success Save" << "\n";
  }
}

static bool detect_marker_black(const cv::Mat& frame_bgr, const cv::Rect& marker_roi, int black_thresh, double black_ratio_thr, int min_black_pixels) {
  if (frame_bgr.empty()) return false;

  cv::Rect full(0,0,frame_bgr.cols, frame_bgr.rows);
  cv::Rect r = marker_roi & full;
  if (r.width <= 0 || r.height <= 0) return false;

  cv::Mat roi = frame_bgr(r);

  cv::Mat gray;
  if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
  else gray = roi;

  cv::Mat black;
  cv::threshold(gray, black, black_thresh, 255, cv::THRESH_BINARY_INV);

  int black_pixels = cv::countNonZero(black);
  double ratio = (double)black_pixels / (double)black.total();

  return (black_pixels >= min_black_pixels) && (ratio >= black_ratio_thr);
}

int main(int argc, char** argv) {
  int cam_index = 0;
  int width = 0, height = 0;

  int show_mode = 2;

  int save_on_defect = 1;
  std::string save_dir = "out_cam";

  bool record_raw = true;         
  double record_fps_hint = 30.0;   
  std::string raw_basename = "raw_input"; 

  int ng_confirm_frames = 3;
  int ok_reset_frames   = 10;

  int use_center_roi = 1;
  double center_w_ratio = 0.5;
  double center_h_ratio = 0.5;

  int marker_need_frames = 3;
  int marker_lost_reset  = 5;

  int marker_black_thresh = 80;
  double marker_black_ratio = 0.01;
  int marker_min_black_px = 200;


  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag)->std::string {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << flag << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };

    if (a == "--cam") cam_index = std::stoi(need(a));
    else if (a == "--w") width = std::stoi(need(a));
    else if (a == "--h") height = std::stoi(need(a));
    else if (a == "--save_dir") save_dir = need(a);
  }

  fs::create_directories(save_dir);
  std::cout << "Save dir: " << fs::absolute(save_dir) << "\n";
  std::cout << "Keys: q/ESC quit \n";

  DetectorParams p;
  p.blur_ksize = 3;
  p.morph_ksize = 9;
  p.area_min = 5.0;
  p.area_max = 1e9;
  p.circ_min = 0.0;

  p.roi = cv::Rect(); 
  DefectDetector det(p);
  cv::Rect last_applied_roi = p.roi;

  cv::VideoCapture cap(cam_index);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open camera index " << cam_index << "\n";
    return 1;
  }
  if (width > 0)  cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
  if (height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

  cv::namedWindow("input", cv::WINDOW_NORMAL);
  cv::namedWindow("vis", cv::WINDOW_NORMAL);
  if (show_mode >= 2) cv::namedWindow("mask", cv::WINDOW_NORMAL);

  cv::VideoWriter raw_writer;
  bool raw_writer_ready = false;
  std::string raw_video_path;

  double fps_ema = 0.0;
  const double alpha = 0.1;

  // NG/OK state 
  int ng_streak = 0;
  int ok_streak = 0;
  bool ng_latched = false;

  // Marker state 
  bool inspect_enable = false;
  int marker_seen_streak = 0;
  int marker_lost_streak = 0;
  bool marker_now = false;

  auto reset_event_state = [&](){
    ng_streak = 0;
    ok_streak = 0;
    ng_latched = false;
  };

  while (true) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "Camera read failed.\n";
      break;
    }

    if (record_raw && !raw_writer_ready) {
      double cam_fps = cap.get(cv::CAP_PROP_FPS);
      double use_fps = (cam_fps > 1.0 && cam_fps < 240.0) ? cam_fps : record_fps_hint;

      std::string ts = ts_ms_string();
      raw_video_path = save_dir + "/" + raw_basename + "_" + ts + ".avi";

      int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
      raw_writer.open(raw_video_path, fourcc, use_fps, frame.size(), true);

      if (!raw_writer.isOpened()) {
        std::cerr << "[WARN] Failed to open RAW video writer: " << raw_video_path << "\n";
        record_raw = false;
      } else {
        raw_writer_ready = true;
        std::cout << "[REC] RAW recording ON: " << fs::absolute(raw_video_path)
                  << "  (fps=" << use_fps << ")\n";
      }
    }

    if (record_raw && raw_writer_ready) {
      raw_writer.write(frame);
    }

    // center ROI 
    cv::Rect center_roi = make_center_roi(frame.cols, frame.rows, center_w_ratio, center_h_ratio);

    cv::Rect applied_roi;
    std::string roi_label;
    if (use_center_roi) {
      applied_roi = center_roi;
      roi_label = "DETECTION ROI (CENTER)";
    } 

    // ROI 반영
    if (!rect_equal(applied_roi, last_applied_roi)) {
      p.roi = applied_roi;
      det = DefectDetector(p);
      last_applied_roi = applied_roi;
    }

    // marker position
    int mw = std::max(60, (int)(frame.cols * 0.20));
    int mh = std::max(60, (int)(frame.rows * 0.20));
    cv::Rect marker_roi(frame.cols - mw, frame.rows - mh, mw, mh);
    marker_now = detect_marker_black(frame, marker_roi, 
                                     marker_black_thresh,
                                     marker_black_ratio,
                                     marker_min_black_px);

    if (marker_now) {
      marker_seen_streak++;
      marker_lost_streak = 0;
    } else {
      marker_lost_streak++;
      marker_seen_streak = 0;
    }
    
    // state 
    bool preinspect = (marker_seen_streak > 0) || inspect_enable;
    if (!inspect_enable && marker_seen_streak >= std::max(1, marker_need_frames)) {
      inspect_enable = true;
      std::cout << "INSPECT ON (marker seen)\n";
    }

    if (inspect_enable && marker_lost_streak >= std::max(1, marker_lost_reset)) {
      inspect_enable = false;
      reset_event_state();
      std::cout << "INSPECT OFF (marker lost)\n";
    }
  
    cv::Mat mask_full = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    std::vector<DefectCandidate> cands;

    double dt=0.0;

    if (preinspect) {
      double t0 = now_ms();
      cands = det.run(frame, mask_full);

      // except of (marker roi & cand bbox)
      cands.erase(
        std::remove_if(cands.begin(), cands.end(),
          [&](const DefectCandidate& c){ return ((c.bbox & marker_roi).area() > 0); }),
        cands.end()
      );
      double t1 = now_ms();
      dt = std::max(0.001, t1 - t0);
      double fps = 1000.0 / dt;
      fps_ema = (fps_ema <= 0.0) ? fps : (fps_ema * (1.0 - alpha) + fps * alpha);
    } else {
      fps_ema=0.0;
    }

    bool ng_now = preinspect ? (!cands.empty()) : false;

    if (preinspect) {
      ng_streak = ng_now ? (ng_streak + 1) : 0;

      if (!ng_now) {
        ok_streak++;
        if (ok_streak >= std::max(1, ok_reset_frames)) {
          ng_latched = false;
        }
      } else {
        ok_streak = 0;
      }
    } else {
      reset_event_state();
    }

    // visualization
    cv::Mat vis_ui = frame.clone();

    for (const auto& cand : cands) {
      cv::rectangle(vis_ui, cand.bbox, cv::Scalar(0, 0, 255), 2);
      char buf[128];
      std::snprintf(buf, sizeof(buf), "A=%.0f C=%.2f", cand.area, cand.circularity);
      cv::putText(vis_ui, buf, cand.bbox.tl() + cv::Point(0, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    // 카메라에서 ROI 영역 표시
    cv::Rect roi_for_ui = (last_applied_roi.area() > 0)
      ? last_applied_roi
      : cv::Rect(0,0,frame.cols,frame.rows);

    if (roi_for_ui.area() > 0) {
      cv::rectangle(vis_ui, roi_for_ui, cv::Scalar(255, 0, 0), 2);
      cv::putText(vis_ui, roi_label, cv::Point(roi_for_ui.x, std::max(20, roi_for_ui.y - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    }

    // marker
    cv::rectangle(vis_ui, marker_roi, cv::Scalar(0, 255, 255), 2);
    cv::putText(vis_ui, std::string("MARKER=") + (marker_now ? "YES" : "NO") + "  PRE=" + (preinspect ? "ON" : "OFF") + "  INSPECT=" + (inspect_enable ? "ON" : "OFF"), cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);

    // final confirm
    if (inspect_enable && save_on_defect && !ng_latched &&
        ng_streak >= std::max(1, ng_confirm_frames)) {
      std::string ts = ts_ms_string();
      save_snapshot_set(save_dir, "defect_" + ts, frame, vis_ui, mask_full);
      ng_latched = true;
    }

    // windows show
    if (show_mode >= 1) cv::imshow("input", frame);
    if (show_mode >= 2) cv::imshow("mask", mask_full);
    cv::imshow("vis", vis_ui);

    // key input
    int key = cv::waitKey(1);
    if (key < 0) continue;
    key &= 0xFF;

    if (key == 27 || key == 'q') break;

    if (key == '2') {
      show_mode = (show_mode + 1) % 3;
      if (show_mode < 2) cv::destroyWindow("mask");
      else cv::namedWindow("mask", cv::WINDOW_NORMAL);
    }
  }

  // stopped writing, camera
  if (raw_writer.isOpened()) raw_writer.release();
  if (cap.isOpened()) cap.release();
  cv::destroyAllWindows();

  return 0;
}
