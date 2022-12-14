#ifndef _POLUGIIN_H
#define _POLUGIIN_H

typedef float PDtype;

extern "C" void get_mt_output(const PDtype* mt_reg_data, const PDtype* mt_cls_data,
                              const int num, const int reg_channels, const int cls_channels,
                              const int height, const int width,
                              PDtype* reg_buffer, PDtype* cls_buffer);
extern "C" void write_img_data_to_buffer(cv::cuda::PtrStepSz<uchar3> img, PDtype *buffer,
                                         const std::vector<PDtype> mean_values, bool to_rgb);

#endif