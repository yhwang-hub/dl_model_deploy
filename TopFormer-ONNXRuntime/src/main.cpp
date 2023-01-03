#include "../include/Topformer.h"

int main(int argc, char** argv)
{
    Net_config cfg = { 0.5, 0.85, "../TopFormer-S_512x512_2x8_160k_sim.onnx" };

    Topformer_detector net(cfg);
	std::string imgpath = "../demo.jpg";
	cv::Mat srcimg = cv::imread(imgpath);
	net.do_detection(srcimg);

    return 0;
}