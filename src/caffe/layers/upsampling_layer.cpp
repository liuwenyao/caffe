#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upsampling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UpsamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
     UpsamplingParameter upsampl_param = this->layer_param_.upsampling_param();
     if(upsampl_param.has_scalefactor()){
          scalefactor_ = upsampl_param.scalefactor();
          CHECK_GT(scalefactor_, 1) << "Scalefactor should be greater than one";
     }
}
template <typename Dtype>
void UpsamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  unpooled_height_ = height_ * scalefactor_;
  unpooled_width_ = width_ * scalefactor_;

  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
      unpooled_width_);
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  caffe_set(top_count, Dtype(0), top_data);
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < unpooled_height_; ++ph) {
        for (int pw = 0; pw < unpooled_width_; ++pw) {
          int sh = ph / scalefactor_;
          int sw = pw / scalefactor_;
          const int index = sh * width_ + sw;
          const int unpooled_index = ph * unpooled_width_ + pw;
          top_data[unpooled_index] = bottom_data[index];
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different unpooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < unpooled_height_; ++ph) {
        for (int pw = 0; pw < unpooled_width_; ++pw) {
          int sh = ph / scalefactor_;
          int sw = pw / scalefactor_;
          const int index = sh * width_ + sw;
          const int unpooled_index = ph * unpooled_width_ + pw;
          bottom_diff[index] += top_diff[unpooled_index];
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
     LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode.";
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top)
{
     LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode.";
}
#endif

#ifdef CPU_ONLY
STUB_GPU(UpsamplingLayer);
#endif

INSTANTIATE_CLASS(UpsamplingLayer);
REGISTER_LAYER_CLASS(Upsampling);

}  // namespace caffe
