#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Rect {
    int x;
    int y;
    int width;
    int height;
}

struct BBox {
    Rect box;
    float confidence;
    int index;
};
 
float get_iou(Rect rect1, Rect rect2) {
    int xx1, yy1, xx2, yy2;
 
    xx1 = max(rect1.x, rect2.x);
    yy1 = max(rect1.y, rect2.y);
    xx2 = min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
 
    int insection_width, insection_height;
    insection_width = max(0, xx2 - xx1 + 1);
    insection_height = max(0, yy2 - yy1 + 1);
 
    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;

    return iou;
}

bool comp(const BBox& b1, const BBox& b2) {
    return b1.confidence > b2.confidence
}

void nms_boxes(const vector<Rect>& boxes, 
               const vector<float>& confidences, 
               float nms_thresh, 
               vector<int> &indices) {
    vector<BBox> bboxes;
    for (int i = 0; i < boxes.size(); i++) {
        BBox bbox;
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), comp);

    int updated_size = bboxes.size();
    indices.clear();
    indices.reserve(updated_size);
    for (int i = 0; i < updated_size; i++) {
        indices.push_back(bboxes[i].index);
        for (int j = i + 1; j < updated_size; j++) {
            float iou = get_iou(bboxes[i].box, bboxes[j].box);
            if (iou > nms_thresh) {
                bboxes.erase(bboxes.begin() + j);
                updated_size = bboxes.size();
            }
        }
    }
}