#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multimodal_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MultiModalBatchNormParameter param = this->layer_param_.multimodal_batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(5);

    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));  // scale blobs_[0]
    this->blobs_[1].reset(new Blob<Dtype>(sz));  // bias blobs_[1]
    this->blobs_[2].reset(new Blob<Dtype>(sz));  // mean blobs_[2]
    this->blobs_[3].reset(new Blob<Dtype>(sz));  // variance blobs_[3]

    shared_ptr<Filler<Dtype> > scale_filler(
      GetFiller<Dtype>(this->layer_param_.multimodal_batch_norm_param().scale_filler()));
    scale_filler->Fill(this->blobs_[0].get());	// Filler della scala
    shared_ptr<Filler<Dtype> > bias_filler(
      GetFiller<Dtype>(this->layer_param_.multimodal_batch_norm_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get()); // Filler del bias

    caffe_set(this->blobs_[2]->count(), Dtype(0.),
        this->blobs_[2]->mutable_cpu_data());
    caffe_set(this->blobs_[3]->count(), Dtype(0.),
        this->blobs_[3]->mutable_cpu_data());

    sz[0]=1;
    this->blobs_[4].reset(new Blob<Dtype>(sz));
    iter_ = 0;
  }
}

template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() > 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  int N = bottom[0]->shape(0); // N = batch size
  int NC = N* channels_; // NC = batch size * channels
  int S = bottom[0]->count() / NC;  // S = H*W 

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  inv_variance_.Reshape(sz);
  temp_C_.Reshape(sz);
  
  sz[0] = N;
  ones_N_.Reshape(sz);
  caffe_set(ones_N_.count(), Dtype(1.), ones_N_.mutable_cpu_data());
  sz[0] = channels_;
  ones_C_.Reshape(sz);
  caffe_set(ones_C_.count(), Dtype(1.), ones_C_.mutable_cpu_data());
  sz[0] = S;
  ones_HW_.Reshape(sz);
  caffe_set(ones_HW_.count(), Dtype(1.), ones_HW_.mutable_cpu_data());

  sz[0] = NC;
  temp_NC_.Reshape(sz);

  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
}

//  multicast x[c] into y[.,c,...]
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::multicast_cpu(int N, int C, int S,
      const Dtype *x, Dtype *y ) {
  Blob<Dtype> templ1_NC;
  vector<int> templ1_size;
  templ1_size.push_back(N*C);
  templ1_NC.Reshape(templ1_size);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1., ones_N_.cpu_data(), x, 0., templ1_NC.mutable_cpu_data()); // C= alpha*A*B+beta*C .... temp_NC=1*ones_N*x + 0 * temp_NC
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., templ1_NC.cpu_data(), ones_HW_.cpu_data(), 0., y); // C= alpha*A*B+beta*C .... y = 1*temp_NC*ones_HW + 0*y

	// result equal a matrix where each vector y[s:s+c,h_i*w_i]=x

}

//  multicast x[c] into y[.,c,...]
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::weights_multicast_cpu(int N, int C, int S,
      const Dtype *x, Dtype *y ) {
  Blob<Dtype> templ1_NC;
  vector<int> templ1_size;
  templ1_size.push_back(N*C);
  templ1_NC.Reshape(templ1_size);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1., x, ones_C_.cpu_data(),0., templ1_NC.mutable_cpu_data()); // C= alpha*A*B+beta*C .... temp_NC=1*ones_N*x + 0 * temp_NC
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., templ1_NC.cpu_data(), ones_HW_.cpu_data(), 0., y); // C= alpha*A*B+beta*C .... y = 1*temp_NC*ones_HW + 0*y

	// result equal a matrix where each vector y[s:s+c,h_i*w_i]=x

}

//  y[c] = sum x(.,c,...)
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::compute_sum_per_channel_cpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {

  Blob<Dtype> templ_NC;
  vector<int> templ_size;
  templ_size.push_back(N*C);
  templ_NC.Reshape(templ_size);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, N * C, S, 1., x, ones_HW_.cpu_data(),
      0., templ_NC.mutable_cpu_data());		// C= alpha*A*B+beta*C .... temp_NC=1*x*ones_HW + 0 * temp_NC -> sum over HW
  caffe_cpu_gemv<Dtype>(CblasTrans, N, C , 1., templ_NC.cpu_data(),
      ones_N_.cpu_data(), 0., y);		// C= alpha*A*B+beta*C .... y=1*temp_NC.T*ones_N + 0 * y -> sum over N
	
	// result equal array y with dim C where each row is the channel sum
}

// MM: y[c] = weighted_mean x(.,c,...)
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::compute_mean_per_channel_cpu(int N, int C, int S, 
    const Dtype *x, const Dtype *w, Dtype *y) {
	Blob<Dtype> app;
  	vector<int> app_size;
  	app_size.push_back(N*C*S);
  	app.Reshape(app_size);
        
	Blob<Dtype> templ_C;
  	vector<int> templ_size;
  	templ_size.push_back(C);
  	templ_C.Reshape(templ_size);
	
	weights_multicast_cpu(N,C,S,w,app.mutable_cpu_data()); 				// W \in NCHW
	const Dtype nsw = 1.0/(caffe_cpu_asum(N,w)*S); 					// 1.0/(sum W * H * W)
	caffe_mul(N*C*S, app.cpu_data(), x, app.mutable_cpu_data()); 	

	caffe_cpu_scale(N*C*S, nsw, app.cpu_data(), app.mutable_cpu_data());		//normalized WX
	compute_sum_per_channel_cpu(N,C,S,app.cpu_data(),y);				//sumWX normalized 	
	

}

template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weights = bottom[1]->cpu_data();
  
  if (use_global_stats_) {
    // use global mean/variance
    caffe_copy(C, this->blobs_[2]->cpu_data(), mean_.mutable_cpu_data());
    caffe_copy(C, this->blobs_[3]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    compute_mean_per_channel_cpu(N, C, S, bottom_data, weights, 
        mean_.mutable_cpu_data()); //E[X]
  }
  //  Y = X- EX
  if (bottom[0] != top[0]) {
    caffe_copy(top_size, bottom_data, top_data);
  }
  multicast_cpu(N, C, S, mean_.cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_axpby(top_size, Dtype(-1.), temp_.cpu_data(),
      Dtype(1.), top_data); // Y = X-E[X]
  if (!use_global_stats_) {
    // compute variance E (X-EX)^2
    caffe_powx(top_size, top_data, Dtype(2.), temp_.mutable_cpu_data()); // Y^2

    compute_mean_per_channel_cpu(N, C, S, temp_.cpu_data(), weights,		// var = E[Y^2]
        variance_.mutable_cpu_data());

    // clip variance
    if ((this->phase_ == TRAIN) && (iter_ <= MULTIMODAL_BN_VARIANCE_CLIP_START))
      iter_++;
    if (iter_ > MULTIMODAL_BN_VARIANCE_CLIP_START) {
      // clip from above
      // temp_C_[c] = average_var + gobal_var[c]
      Dtype y = caffe_cpu_asum(C, this->blobs_[3]->cpu_data());
      caffe_cpu_scale(C, Dtype(y/C), ones_C_.cpu_data(),
          temp_C_.mutable_cpu_data());
      caffe_cpu_axpby(C, Dtype(1.0), this->blobs_[3]->cpu_data(),
          Dtype(1.0), temp_C_.mutable_cpu_data());
      caffe_cpu_eltwise_min(C,
          Dtype(MULTIMODAL_BN_VARIANCE_CLIP_CONST), temp_C_.cpu_data(),
          Dtype(1.0), variance_.mutable_cpu_data());
      // clip from below
      caffe_cpu_eltwise_max(C,
          Dtype((1.)/MULTIMODAL_BN_VARIANCE_CLIP_CONST), this->blobs_[3]->cpu_data(),
          Dtype(1.0), variance_.mutable_cpu_data());
    }
    //  update global mean and variance
    if (iter_ > 1) {
      caffe_cpu_axpby(C,
          Dtype(1. - moving_average_fraction_), mean_.cpu_data(),
          Dtype(moving_average_fraction_), this->blobs_[2]->mutable_cpu_data());
      caffe_cpu_axpby(C,
          Dtype(1.- moving_average_fraction_), variance_.cpu_data(),
          Dtype(moving_average_fraction_), this->blobs_[3]->mutable_cpu_data());
    } else {
      caffe_copy(C, mean_.cpu_data(), this->blobs_[2]->mutable_cpu_data());
      caffe_copy(C, variance_.cpu_data(), this->blobs_[3]->mutable_cpu_data());
    }
  }

  //  inv_var= ( eps+ variance)^(-0.5)
  caffe_add_scalar(C, eps_, variance_.mutable_cpu_data()); // var + eps
  
  caffe_powx(C, variance_.cpu_data(), Dtype(-0.5),
      inv_variance_.mutable_cpu_data()); // std = (var+eps)^(-0.5)
  // X_norm = (X-EX) * inv_var
  multicast_cpu(N, C, S, inv_variance_.cpu_data(), temp_.mutable_cpu_data()); // repeat std
  caffe_mul(top_size, top_data, temp_.cpu_data(), top_data); // (X-E[X])/std 

  weights_multicast_cpu(N,C,S,weights,temp_.mutable_cpu_data());

  caffe_mul(top_size, top_data, temp_.cpu_data(), top_data); // (X-E[X])/std * w


  // copy x_norm for backward
  caffe_copy(top_size, top_data, x_norm_.mutable_cpu_data());
  
  // -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
  // Y = X_norm * scale[c]
  const Blob<Dtype> & scale_data = *(this->blobs_[0]);
  multicast_cpu(N, C, S, scale_data.cpu_data(), temp_.mutable_cpu_data()); // repeat gamma
  caffe_mul(top_size, top_data, temp_.cpu_data(), top_data); // X_norm * gamma
  // Y = Y + shift[c]
  const Blob<Dtype> & shift_data = *(this->blobs_[1]);
  multicast_cpu(N, C, S, shift_data.cpu_data(), temp_.mutable_cpu_data());	// repeat beta
 
  caffe_add(top_size, top_data, temp_.mutable_cpu_data(), top_data);		// + bias

}




template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom){
	//TODO   now just a placeholder
	for (int i = 0; i < bottom.size(); ++i) {
    		if (propagate_down[i]) {
      			caffe_set(bottom[i]->count(), Dtype(0),
                	bottom[i]->mutable_cpu_diff());
		} 
	}
}

#ifdef CPU_ONLY
STUB_GPU(MultiModalBatchNormLayer);
#endif

INSTANTIATE_CLASS(MultiModalBatchNormLayer);

}  // namespace caffe
