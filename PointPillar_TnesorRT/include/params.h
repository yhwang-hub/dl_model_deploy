#ifndef _PARAMS_H
#define _PARAMS_H

const int MAX_VOXELS = 40000;
class Params
{
  public:
    static const int num_classes = 3;
    const char *class_name [num_classes] = { "Car","Pedestrian","Cyclist",};
    const float min_x_range = 0.0;
    const float max_x_range = 69.12;
    const float min_y_range = -39.68;
    const float max_y_range = 39.68;
    const float min_z_range = -3.0;
    const float max_z_range = 1.0;
    // the size of a pillar
    const float pillar_x_size = 0.16;
    const float pillar_y_size = 0.16;
    const float pillar_z_size = 4.0;
    const int max_num_points_per_pillar = 32;
    const int num_point_values = 4;
    // the number of feature maps for pillar scatter
    const int num_feature_scatter = 64;
    const float dir_offset = 0.78539;
    const float dir_limit_offset = 0.0;
    // the num of direction classes(bins)
    const int num_dir_bins = 2;
    // anchors decode by (x, y, z, dir)
    static const int num_anchors = num_classes * 2;
    static const int len_per_anchor = 4;
    const float anchors[num_anchors * len_per_anchor] = {
      3.9,1.6,1.56,0.0,
      3.9,1.6,1.56,1.57,
      0.8,0.6,1.73,0.0,
      0.8,0.6,1.73,1.57,
      1.76,0.6,1.73,0.0,
      1.76,0.6,1.73,1.57,
      };
    const float anchor_bottom_heights[num_classes] = {-1.78,-0.6,-0.6,};
    // the score threshold for classification
    const float score_thresh = 0.1;
    const float nms_thresh = 0.01;
    const int max_num_pillars = MAX_VOXELS;
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;
    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;
    // the input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
    // the output size of the 2D backbone network
    const int feature_x_size = grid_x_size / 2;
    const int feature_y_size = grid_y_size / 2;
    Params() {};
};

#endif