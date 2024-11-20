/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : model_run.cc
 * Authors     : klyu
 * Create Time : 2020-10-28 12:22:44 (CST)
 * Description :
 *
 */

#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "./stb/stb_image_resize.h"
static const uint8_t color[3] = {0xff, 0, 0};

#include "inference_nv12.h"
#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstring>
#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)


using namespace std;
using namespace magik::venus;

extern std::unique_ptr<venus::BaseNet> yolo_net;
extern std::unique_ptr<venus::BaseNet> face_net;

void write_output_bin(const float* out_ptr, int size)
{
    std::string out_name = "out_res.bin";
    std::ofstream owput;
    owput.open(out_name, std::ios::binary);
    if (!owput || !owput.is_open() || !owput.good()) {
        owput.close();
        return ;
    }
    owput.write((char *)out_ptr, size * sizeof(float));
    owput.close();
    return ;
}


typedef struct
{
    unsigned char* image;  
    int w;
    int h;
}input_info_t;

struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};


void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

void trans_coords(std::vector<magik::venus::ObjBbox_t> &in_boxes, PixelOffset &pixel_offset,float scale){
    
    // printf("pad_x:%d pad_y:%d scale:%f \n",pixel_offset.left,pixel_offset.top,scale);
    for(int i = 0; i < in_boxes.size(); i++) {

        in_boxes[i].box.x0 = (in_boxes[i].box.x0 - pixel_offset.left) / scale;
        in_boxes[i].box.x1 = (in_boxes[i].box.x1 - pixel_offset.left) / scale;
        in_boxes[i].box.y0 = (in_boxes[i].box.y0 - pixel_offset.top) / scale;
        in_boxes[i].box.y1 = (in_boxes[i].box.y1 - pixel_offset.top) / scale;
    }
}


void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h);
void manyclass_nms(std::vector<magik::venus::ObjBbox_t> &input, std::vector<magik::venus::ObjBbox_t> &output, int classnums, int type, float nms_threshold);


vector<vector<float>> min_boxes = {{10.0, 16.0, 24.0}, {32.0, 48.0}, {64.0, 96.0}, {128.0, 192.0, 256.0}};
vector<float> strides = {8.0, 16.0, 32.0, 64.0};

vector<vector<float>> generate_priors(const vector<vector<int>>& feature_map_list, const vector<vector<float>>& shrinkage_list, const vector<int>& image_size, const vector<vector<float>>& min_boxes) {
    vector<vector<float>> priors;
    for (size_t index = 0; index < feature_map_list[0].size(); ++index) {
        float scale_w = image_size[0] / shrinkage_list[0][index];
        float scale_h = image_size[1] / shrinkage_list[1][index];
        for (int j = 0; j < feature_map_list[1][index]; ++j) {
            for (int i = 0; i < feature_map_list[0][index]; ++i) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float min_box : min_boxes[index]) {
                    float w = min_box / image_size[0];
                    float h = min_box / image_size[1];
                    priors.push_back({x_center, y_center, w, h});
                }
            }
        }
    }
    // cout << "priors nums:" << priors.size() << endl;
    // Clipping the priors to be within [0.0, 1.0]
    for (auto& prior : priors) {  
        for (auto& val : prior) {  
            val = std::min(std::max(val, 0.0f), 1.0f);  
        }  
    }  

    return priors;
}

vector<vector<float>> define_img_size(const vector<int>& image_size) {
    vector<vector<int>> feature_map_w_h_list;
    vector<vector<float>> shrinkage_list;
    for (int size : image_size) {
        vector<int> feature_map;
        for (float stride : strides) {
            feature_map.push_back(static_cast<int>(ceil(size / stride)));
        }
        feature_map_w_h_list.push_back(feature_map);
    }

    for (size_t i = 0; i < image_size.size(); ++i) {
        shrinkage_list.push_back(strides);
    }
    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes);
}

vector<vector<float>> convert_locations_to_boxes(const vector<vector<float>>& locations, const vector<vector<float>>& priors, float center_variance, float size_variance) {
    vector<vector<float>> boxes;
    for (size_t i = 0; i < locations.size(); ++i) {
        vector<float> box;
        for (size_t j = 0; j < locations[i].size() / 4; ++j) {
            float cx = locations[i][j * 4 + 0] * center_variance * priors[i][2] + priors[i][0];
            float cy = locations[i][j * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(locations[i][j * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(locations[i][j * 4 + 3] * size_variance) * priors[i][3];
            box.push_back(cx);
            box.push_back(cy);
            box.push_back(w);
            box.push_back(h);
        }
        boxes.push_back(box);
    }
    return boxes;
}

vector<vector<float>> center_form_to_corner_form(const vector<vector<float>>& locations) {
    vector<vector<float>> boxes;
    for (const auto& loc : locations) {
        vector<float> box;
        for (size_t i = 0; i < loc.size() / 4; ++i) {
            float cx = loc[i * 4 + 0];
            float cy = loc[i * 4 + 1];
            float w = loc[i * 4 + 2];
            float h = loc[i * 4 + 3];
            float xmin = cx - w / 2;
            float ymin = cy - h / 2;
            float xmax = cx + w / 2;
            float ymax = cy + h / 2;
            box.push_back(xmin);
            box.push_back(ymin);
            box.push_back(xmax);
            box.push_back(ymax);
        }
        boxes.push_back(box);
    }
    return boxes;
}

float area_of(float left, float top, float right, float bottom) {
    float width = max(0.0f, right - left);
    float height = max(0.0f, bottom - top);
    return width * height;
}

float iou_of(const vector<float>& box0, const vector<float>& box1) {
    float overlap_left = max(box0[0], box1[0]);
    float overlap_top = max(box0[1], box1[1]);
    float overlap_right = min(box0[2], box1[2]);
    float overlap_bottom = min(box0[3], box1[3]);

    float overlap_area = area_of(overlap_left, overlap_top, overlap_right, overlap_bottom);
    float area0 = area_of(box0[0], box0[1], box0[2], box0[3]);
    float area1 = area_of(box1[0], box1[1], box1[2], box1[3]);
    float total_area = area0 + area1 - overlap_area;
    if (total_area <= 0.0f) return 0.0f;
    return overlap_area / total_area;
}

vector<vector<float>> hard_nms(const vector<vector<float>>& box_scores, float iou_threshold, int top_k = -1, int candidate_size = 200) {
    vector<int> idx(box_scores.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&box_scores](int i1, int i2) {
        return box_scores[i1].back() < box_scores[i2].back();
    });

    if (candidate_size > 0 && candidate_size < (int)idx.size()) {
        idx.resize(candidate_size);
    }

    vector<vector<float>> picked;
    while (!idx.empty()) {
        int current = idx.back();
        const auto& current_box = box_scores[current];
        picked.push_back(current_box);
        if (top_k > 0 && (int)picked.size() >= top_k) break;
        idx.pop_back();

        for (auto it = idx.begin(); it != idx.end();) {
            float iou = iou_of(box_scores[*it], current_box);
            if (iou > iou_threshold) {
                it = idx.erase(it);
            } else {
                ++it;
            }
        }
    }
    return picked;
}

vector<vector<float>> predict(float width, float height, const vector<vector<float>>& scores, const vector<vector<float>>& boxes, float prob_threshold, float iou_threshold = 0.3, int top_k = -1) {
    vector<vector<float>> final_boxes;
    vector<vector<float>> box_scores; // Combine boxes and scores in the required format
    for (size_t i = 0; i < boxes.size(); ++i) {
        vector<float> box_score = boxes[i];
        box_score.push_back(scores[i][1]); // Assuming class score is at index 1
        if (scores[i][1] > prob_threshold) {
            box_scores.push_back(box_score);
        }
    }
    
    vector<vector<float>> picked = hard_nms(box_scores, iou_threshold, top_k);

    // Convert coordinates back to original scale and print
    for (const auto& box : picked) {
        // cout << "Box: ";
        vector<float> face_box;
        for (size_t i = 0; i < 4; ++i) {
            float coord = i % 2 == 0 ? box[i] * width : box[i] * height;
            face_box.push_back((int)coord);
            // cout << coord << " ";
        }
        final_boxes.push_back(face_box);
        // cout << "Score: " << box.back() << endl;
    }
    return final_boxes;
}

void softmax(const float* input, float* output, int w, int h, int c) {
    const float* in_data = input;
    int first = h;
    int second = c;
    int third = w;

    int softmax_size = w * h;
    float* softmax_data = (float*)malloc(softmax_size * sizeof(float));
    float* max = (float*)malloc(softmax_size * sizeof(float));
    if (softmax_data == NULL || max == NULL) {  
        // Handle memory allocation failure  
        if (softmax_data) free(softmax_data);  
        if (max) free(max);  
        return;  
    } 
    for (int f = 0; f < first; ++f) {
        for (int t = 0; t < third; ++t) {
            int m_under = f * third + t;
            max[m_under] = -FLT_MAX;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                max[m_under] = in_data[i_under] > max[m_under] ? in_data[i_under] : max[m_under];
            }
            softmax_data[m_under] = 0;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float temp = in_data[i_under];
                softmax_data[m_under] += exp(temp - max[m_under]);
            }
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float input_num = in_data[i_under];
                float softmax_num = exp(input_num - max[m_under]) / softmax_data[m_under];
                output[i_under] = softmax_num;
            }
        }
    }
    // Free the allocated memory  
    free(softmax_data);  
    free(max);
}



int Goto_Magik_Detect(char * nv12Data, int width, int height)
{
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            printf("warning: could not set CPU affinity, continuing...\n");
    }


    void *handle = NULL;

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int face_in_w = 320, face_in_h = 240;
    int yolo_in_w = 640, yolo_in_h = 384;
    // int yolo_in_w = 640, yolo_in_h = 640;
    
    PixelOffset pixel_offset;
 

	std::unique_ptr<venus::Tensor> input;
    std::unique_ptr<venus::Tensor> face_input;

    input_info_t input_src;
    //
    input_src.w = width;
    input_src.h = height;
    input_src.image = (unsigned char*)nv12Data;
    
    //---------------------process-------------------------------
    // get ori image w h
    ori_img_w = input_src.w;
    ori_img_h = input_src.h;

    int line_stride = ori_img_w;
    input = yolo_net->get_input(0);
    
    input->reshape({1, yolo_in_h, yolo_in_w, 1});

    uint8_t *indata = input->mudata<uint8_t>();
    

    //resize and padding
    magik::venus::Tensor temp_ori_input({1, ori_img_h, ori_img_w, 1}, TensorFormat::NV12);
    uint8_t *tensor_data = temp_ori_input.mudata<uint8_t>();
    int src_size = int(ori_img_h * ori_img_w * 1.5);
    magik::venus::memcopy((void*)tensor_data, (void*)input_src.image, src_size * sizeof(uint8_t));

    float scale_x = (float)yolo_in_w/(float)ori_img_w;
    float scale_y = (float)yolo_in_h/(float)ori_img_h;
    scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    // printf("scale---> %f\n",scale);

    int valid_dst_w = (int)(scale*ori_img_w);
    if (valid_dst_w % 2 == 1)
        valid_dst_w = valid_dst_w + 1;
    int valid_dst_h = (int)(scale*ori_img_h);
    if (valid_dst_h % 2 == 1)
        valid_dst_h = valid_dst_h + 1;

    int dw = yolo_in_w - valid_dst_w;
    int dh = yolo_in_h - valid_dst_h;

    pixel_offset.top = int(round(float(dh)/2 - 0.1));
    pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    pixel_offset.left = int(round(float(dw)/2 - 0.1));
    pixel_offset.right = int(round(float(dw)/2 + 0.1));
    
//    check_pixel_offset(pixel_offset);

    magik::venus::BsCommonParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.input_height = ori_img_h;
    param.input_width = ori_img_w;
    param.input_line_stride = ori_img_w;
    param.in_layout = magik::venus::ChannelLayout::NV12;
    param.out_layout = magik::venus::ChannelLayout::NV12;

    magik::venus::common_resize((const void*)tensor_data, *input.get(), magik::venus::AddressLocate::NMEM_VIRTUAL, &param);
    
    yolo_net->run();

    std::unique_ptr<const venus::Tensor> out0 = yolo_net->get_output(0);
    std::unique_ptr<const venus::Tensor> out1 = yolo_net->get_output(1);
    std::unique_ptr<const venus::Tensor> out2 = yolo_net->get_output(2);

    auto shape0 = out0->shape();
    auto shape1 = out1->shape();
    auto shape2 = out2->shape();

    int shape_size0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int shape_size1 = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    int shape_size2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    venus::Tensor temp0(shape0);
    venus::Tensor temp1(shape1);
    venus::Tensor temp2(shape2);

    float* p0 = temp0.mudata<float>();
    float* p1 = temp1.mudata<float>();
    float* p2 = temp2.mudata<float>();

    memcopy((void*)p0, (void*)out0->data<float>(), shape_size0 * sizeof(float));
    memcopy((void*)p1, (void*)out1->data<float>(), shape_size1 * sizeof(float));
    memcopy((void*)p2, (void*)out2->data<float>(), shape_size2 * sizeof(float));
   
    std::vector<venus::Tensor> out_res;
    out_res.push_back(temp0);
    out_res.push_back(temp1);
    out_res.push_back(temp2);

    std::vector<magik::venus::ObjBbox_t>  output_boxes;
    output_boxes.clear();
    generateBBox(out_res, output_boxes, yolo_in_w, yolo_in_h);
    trans_coords(output_boxes, pixel_offset, scale);

    // for (int i = 0; i < int(output_boxes.size()); i++) {
    //     auto person = output_boxes[i];
    //     printf("Person:   ");
    //     printf("%d ",(int)person.box.x0);
    //     printf("%d ",(int)person.box.y0);
    //     printf("%d ",(int)person.box.x1);
    //     printf("%d ",(int)person.box.y1);
    //     printf("%.2f ",person.score);
    //     printf("\n");
    // }
    if (int(output_boxes.size()) == 0)
    {
        printf("No person detected !\n");
        return 0;
    }

    printf("Person detected, ");
        

    face_input = face_net->get_input(0);
    face_input->reshape({1, face_in_h, face_in_w, 1});
    magik::venus::common_resize((const void*)tensor_data, *face_input.get(), magik::venus::AddressLocate::NMEM_VIRTUAL, &param);
    face_net->run();

     // postprocessing
    std::unique_ptr<const venus::Tensor> out_0 = face_net->get_output(0); 
    std::unique_ptr<const venus::Tensor> out_1 = face_net->get_output(1);

    const float* output_data_0 = out_0->data<float>();
    const float* output_data_1 = out_1->data<float>();

    auto shape_0 = out_0->shape(); // scores
    auto shape_1 = out_1->shape(); // boxes

    int scores_size = shape_0[0]*shape_0[1]*shape_0[2]; // 1,4420,2
    int boxes_size  = shape_1[0]*shape_1[1]*shape_1[2]; // 1,4420,4,

    float* output_data_0_softmax = (float*)malloc(scores_size * sizeof(float));
    softmax(output_data_0, output_data_0_softmax, shape_0[0], shape_0[1], shape_0[2]);

    vector<vector<float>> scores;
    vector<vector<float>> boxes;

    // Assuming shape_0[1] == shape_1[1]: give the number of detections
    for (int i = 0; i < shape_0[1]; ++i) {
        // Extract scores
        vector<float> score;
        for (int j = 0; j < shape_0[2]; ++j) {
            score.push_back(output_data_0_softmax[i * shape_0[2] + j]);
        }
        scores.push_back(score);
  
        // Extract boxes
        vector<float> box;
        // Assuming shape_0[2] == 4, for [x1, y1, x2, y2]
        for (int k = 0; k < shape_1[2]; ++k) {
            box.push_back(output_data_1[i * shape_1[2] + k]);
        }
        boxes.push_back(box);
    }

    free(output_data_0_softmax);

    vector<int> input_size = {320, 240};
    float center_variance = 0.1;
    float size_variance = 0.2;
    float prob_threshold = 0.7;
    float iou_threshold = 0.3;
    int top_k = -1;

    vector<vector<float>> priors = define_img_size(input_size);
    
    vector<vector<float>> converted_boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance);

    vector<vector<float>> final_boxes = center_form_to_corner_form(converted_boxes);

    vector<vector<float>> final_face_boxes = predict(ori_img_w, ori_img_h, scores, final_boxes, prob_threshold, iou_threshold, top_k);

    if (final_face_boxes.size() > 0){
        printf("Face detected\n");
    }
    else {
        printf("Face covered or rolled over !!!\n");
    }

    return 0;
}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h){

  float person_threshold = 0.3;
  int classes = 80;
  float nms_threshold = 0.6;
  std::vector<float> strides = {8.0, 16.0, 32.0};
  int box_num = 3;
  std::vector<float> anchor = {10,13,  16,30,  33,23, 30,61,  62,45,  59,119, 116,90,  156,198,  373,326};


  std::vector<magik::venus::ObjBbox_t>  temp_boxes;
  venus::generate_box(out_res, strides, anchor, temp_boxes, img_w, img_h, classes, box_num, person_threshold, magik::venus::DetectorType::YOLOV5);
//  venus::nms(temp_boxes, candidate_boxes, nms_threshold); 
  manyclass_nms(temp_boxes, candidate_boxes, classes, 0, nms_threshold);

}

void manyclass_nms(std::vector<magik::venus::ObjBbox_t> &input, std::vector<magik::venus::ObjBbox_t> &output, int classnums, int type, float nms_threshold) {
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  std::vector<magik::venus::ObjBbox_t> classbuf;
  for (int clsid = 0; clsid < classnums; clsid++) {
    classbuf.clear();
    for (int i = 0; i < box_num; i++) {
      if (merged[i])
        continue;
      if(clsid!=input[i].class_id)
        continue;
      classbuf.push_back(input[i]);
      merged[i] = 1;

    }
    magik::venus::nms(classbuf, output, nms_threshold, magik::venus::NmsType::HARD_NMS);
  }
}

