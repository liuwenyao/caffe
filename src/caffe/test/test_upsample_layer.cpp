#include <algorithm>
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/upsampling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
template <typename TypeParam>
class UpsamplingLayerTest : public MultiDeviceTest<TypeParam> {
     typedef typename TypeParam::Dtype Dtype;
protected:
     UpsamplingLayerTest()
          : blob_bottom_(new Blob<Dtype>()),
            blob_top_(new Blob<Dtype>())
          {}
     virtual void SetUp() {
          Caffe::set_random_seed(1701);
          blob_bottom_->Reshape(1, 3, 2, 2);
          // fill the values
          FillerParameter filler_param;
          GaussianFiller<Dtype> filler(filler_param);
          filler.Fill(this->blob_bottom_);
          blob_bottom_vec_.push_back(blob_bottom_);
          blob_top_vec_.push_back(blob_top_);
     }
     virtual ~UpsamplingLayerTest()
          {
               delete blob_bottom_;
               delete blob_top_;
          }
     Blob<Dtype>* const blob_bottom_;
     Blob<Dtype>* const blob_top_;
     vector<Blob<Dtype>*> blob_bottom_vec_;
     vector<Blob<Dtype>*> blob_top_vec_;

     void TestForward(){
          LayerParameter layer_param;
          UpsamplingParameter *upsample_param = layer_param.mutable_upsampling_param();
          upsample_param->set_scalefactor(2);

          const int num = 1;
          const int channel = 3;
          blob_bottom_->Reshape(num, channel, 2, 2);
          Dtype *bottom_data = blob_bottom_->mutable_cpu_data();
          for(int i=0; i < num; i++){
               for(int j=0; j < channel; j++){
                    printf("Bottom data: \n");
                    for(int k=0; k < 2; k++){
                         for(int h=0; h < 2; h++){
                              bottom_data[k*2 +h] = k*2 + h;
                              printf("%.2f ", bottom_data[k*2 + h]);
                         }
                         printf("\n");
                    }

                    bottom_data += blob_bottom_->offset(0, 1);
               }
          }
          UpsamplingLayer<Dtype> layer(layer_param);
          layer.SetUp(blob_bottom_vec_, blob_top_vec_);
          EXPECT_EQ(blob_top_->num(), num);
          EXPECT_EQ(blob_top_->channels(), channel);
          EXPECT_EQ(blob_top_->height(), 4);
          EXPECT_EQ(blob_top_->width(), 4);
          layer.Forward(blob_bottom_vec_, blob_top_vec_);

          const Dtype * top_data = blob_top_->cpu_data();
          for(int i=0; i < num; i++){
               for(int j=0; j < channel; j++){
                    printf("Top data: \n");
                    for(int k=0; k < 4; k++){
                         for(int h=0; h < 4; h++){
                              EXPECT_EQ(top_data[k*4 +h],(k/2)*2 + h/2);
                              printf("%.2f ", top_data[k*4 +h]);
                         }
                         printf("\n");
                    }

                    top_data += blob_top_->offset(0, 1);
               }
          }
     }

};

TYPED_TEST_CASE(UpsamplingLayerTest, TestDtypesAndDevices);
TYPED_TEST(UpsamplingLayerTest, TestForward){
      this->TestForward();
}

     TYPED_TEST(UpsamplingLayerTest, PrintBackward) {
          typedef typename TypeParam::Dtype Dtype;

          LayerParameter layer_param;
          UpsamplingParameter *upsample_param = layer_param.mutable_upsampling_param();
          upsample_param->set_scalefactor(2);

          UpsamplingLayer<Dtype> layer(layer_param);
          layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
          cout << "bottom count: " << this->blob_bottom_->count() << endl;
          cout << "top count: " << this->blob_top_->count() << endl;

          const int num = 1;
          const int channel = 3;
          Dtype *top_diff = this->blob_top_->mutable_cpu_diff();
          int idx = 0;
          for(int i=0; i < num; i++){
               for(int j=0; j < channel; j++){
                    printf("Top diff c(%d): \n", j);
                    for(int k=0; k < 4; k++){
                         for(int h=0; h < 4; h++){
                              top_diff[k*4 + h] = idx++;
                              printf("%.2f ", top_diff[k*4 +h]);
                         }
                         printf("\n");
                    }

                    top_diff += this->blob_top_->offset(0, 1);
               }
          }
          const std::vector<bool> prop(1, true);
          layer.Backward(this->blob_top_vec_, prop, this->blob_bottom_vec_);
          float ans[] = {10, 18, 42, 50, 74, 82, 106, 114, 138, 146, 170, 178};
          for(int i=0; i < this->blob_bottom_->count(); i++){
               EXPECT_EQ(this->blob_bottom_->cpu_diff()[i], ans[i]);
          }
          const Dtype *bottom_diff = this->blob_bottom_->cpu_diff();
          for(int i=0; i < num; i++){
               for(int j=0; j < channel; j++){
                    printf("bottom diff(%d): \n", j);
                    for(int k=0; k < 2; k++){
                         for(int h=0; h < 2; h++){
                              printf("%.2f ", bottom_diff[k*2 +h]);
                         }
                         printf("\n");
                    }

                    bottom_diff += this->blob_bottom_->offset(0, 1);
               }
          }
     }

     TYPED_TEST(UpsamplingLayerTest, TestSetup){
          typedef typename TypeParam::Dtype Dtype;
          LayerParameter layer_param;
          UpsamplingParameter* upsample_param = layer_param.mutable_upsampling_param();
          upsample_param->set_scalefactor(2);
          UpsamplingLayer<Dtype> layer(layer_param);
          layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
          EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
          EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
          EXPECT_EQ(this->blob_top_->height(), 4);
          EXPECT_EQ(this->blob_top_->width(), 4);
     }
}// namespace caffe
