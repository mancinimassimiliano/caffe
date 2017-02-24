#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include <math.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multimodal_batch_norm_layer.hpp"
//#include "caffe/layers/cudnn_batch_norm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 20
#define INPUT_DATA_SIZE 16

namespace caffe {

  template <typename TypeParam>
  class MultiModalBatchNormLayerTest : public MultiDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;
protected:
  MultiModalBatchNormLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(20, 16, 10, 10)),
        blob_bottom_b_(new Blob<Dtype>(20, 1,1,1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    //unifiller.value()=0.5;
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);

  }
  virtual ~MultiModalBatchNormLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


  TYPED_TEST_CASE(MultiModalBatchNormLayerTest, TestDtypesAndDevices);

  TYPED_TEST(MultiModalBatchNormLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    MultiModalBatchNormParameter* mmbn_param = layer_param.mutable_multimodal_batch_norm_param();
    FillerParameter *scale_param = mmbn_param->mutable_scale_filler();
    scale_param->set_value(Dtype(1.0));
    FillerParameter *bias_param = mmbn_param->mutable_bias_filler();
    bias_param->set_value(Dtype(0.0));

    mmbn_param->set_eps(0.2);
    
    Dtype eps = 0.2;
     const Dtype kErrorBound = 0.001;
    
    MultiModalBatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_a_->num();
    int channels = this->blob_bottom_a_->channels();
    int height = this->blob_bottom_a_->height();
    int width = this->blob_bottom_a_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0, sumw=0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_bottom_a_->data_at(i, j, k, l);
	    Dtype w = this->blob_bottom_b_->data_at(i, 0,0,0);
            sum += data*w;
	    sumw += w;
          }
        }
        
      }
      Dtype mean_computed = sum/sumw;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_bottom_a_->data_at(i, j, k, l);
	    Dtype w = this->blob_bottom_b_->data_at(i, 0,0,0);
            var += (data-mean_computed)*(data-mean_computed)*w;
	    
          }
        }
        
      }
      Dtype var_computed = var/sumw;
     
      
      sum = 0; 
      var = 0;
      // Check output correctness
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
	    Dtype in = this->blob_bottom_a_->data_at(i, j, k, l);
	    Dtype w = this->blob_bottom_b_->data_at(i, 0,0,0);
            //EXPECT_NEAR(0, data-w*(in-mean_computed)/sqrt(var_computed), kErrorBound);
	    sum += data;
	    var += data*data;
           EXPECT_NEAR(data, w*(in-mean_computed)/sqrt(var_computed+eps), kErrorBound);
            
          }
        }
        
      }
      
      // expect zero mean
     // EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      //EXPECT_NEAR(1, var, kErrorBound);
    
    
            
	}
  }
  
  


}  // namespace caffe
