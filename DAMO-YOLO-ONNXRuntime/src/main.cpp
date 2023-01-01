#include "../include/damo_yolo.h"

int main(int argc, char** argv)
{
    Net_config cfg = { 0.5, 0.85, "../damoyolo_tinynasL20_T_192x320.onnx" }; ///choices=["weights/damoyolo_tinynasL20_T_192x320.onnx", "weights/damoyolo_tinynasL20_T_256x320.onnx", "weights/damoyolo_tinynasL20_T_256x416.onnx", "weights/damoyolo_tinynasL20_T_288x480.onnx", "weights/damoyolo_tinynasL20_T_384x640.onnx", "weights/damoyolo_tinynasL20_T_480x640.onnx", "weights/damoyolo_tinynasL20_T_480x800.onnx", "weights/damoyolo_tinynasL20_T_640x640.onnx", "weights/damoyolo_tinynasL20_T_736x1280.onnx", "weights/damoyolo_tinynasL25_S_192x320.onnx", "weights/damoyolo_tinynasL25_S_256x320.onnx", "weights/damoyolo_tinynasL25_S_256x416.onnx", "weights/damoyolo_tinynasL25_S_288x480.onnx", "weights/damoyolo_tinynasL25_S_384x640.onnx", "weights/damoyolo_tinynasL25_S_480x640.onnx", "weights/damoyolo_tinynasL25_S_480x800.onnx", "weights/damoyolo_tinynasL25_S_640x640.onnx", "weights/damoyolo_tinynasL25_S_736x1280.onnx", "weights/damoyolo_tinynasL35_M_192x320.onnx", "weights/damoyolo_tinynasL35_M_256x320.onnx", "weights/damoyolo_tinynasL35_M_256x416.onnx", "weights/damoyolo_tinynasL35_M_288x480.onnx", "weights/damoyolo_tinynasL35_M_384x640.onnx", "weights/damoyolo_tinynasL35_M_480x640.onnx", "weights/damoyolo_tinynasL35_M_480x800.onnx", "weights/damoyolo_tinynasL35_M_640x640.onnx", "weights/damoyolo_tinynasL35_M_736x1280.onnx"]
	damo_yolo_detector net(cfg);
	std::string imgpath = "../dog.jpg";
	cv::Mat srcimg = cv::imread(imgpath);
	net.do_detection(srcimg);

	static const std::string kWinName = "Deep learning object detection in ONNXRuntime";
	cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
	cv::imshow(kWinName, srcimg);
	cv::waitKey(0);
	cv::destroyAllWindows();

    return 0;
}