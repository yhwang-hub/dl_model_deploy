#include "../include/BYTETracker.h"
#include "../include/yolox.h"

int main(int argc, char** argv) {
    // string(argv[2]) == "-i"
    const string engine_file_path(argv[1]);
    const string input_video_path{argv[3]};

    Yolox_Detector* yolox_instance = new Yolox_Detector(engine_file_path);

    VideoCapture cap(input_video_path);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;

    VideoWriter writer("demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    Mat img;
    std::cout<<"fps:"<<fps<<std::endl;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 0;
	while (true)
    {
        if(!cap.read(img))
            break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            std::cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << std::endl;
        }
		if (img.empty())
			break;

        auto start = chrono::system_clock::now();
        yolox_instance->do_detection(img);
        std::vector<Object> objects = yolox_instance->getDetectResults();
        std::cout<<"after detection Num of boxes: "<< objects.size() <<std::endl;
        std::vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms +=chrono::duration_cast<chrono::microseconds>(end - start).count();

        std::cout<<"after track Num of boxes:"<<output_stracks.size()<<std::endl;

        for (int i = 0; i < output_stracks.size(); i++)
		{
			std::vector<float> tlwh = output_stracks[i].tlwh;
            std::cout<<"tracking, x1:"<<tlwh[0]
                    <<", y1:"<<tlwh[1]
                    <<", x2:"<<tlwh[2]
                    <<", y2:"<<tlwh[3]
                    <<std::endl;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			// if (tlwh[2] * tlwh[3] > 20 && !vertical)
			// {
			// 	Scalar s = tracker.get_color(output_stracks[i].track_id);
			// 	cv::putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
            //             0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            //     cv::rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			// }
            Scalar s = tracker.get_color(output_stracks[i].track_id);
            cv::putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                    0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            cv::rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
		}
        cv::putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
    }
    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;
    if(yolox_instance)
    {
        delete yolox_instance;
        yolox_instance = nullptr;
    }
    return 0;
}
