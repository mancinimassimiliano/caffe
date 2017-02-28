#include <algorithm>
#include <vector>

#include "caffe/layers/multimodal_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// multicast x[c] into y[.,c,...]
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::multicast_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1., ones_N_.gpu_data(), x, 0., temp_NC.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., temp_NC.gpu_data(), ones_HW_.gpu_data(), 0., y);
}


// multicast x[c] into y[.,c,...]
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::weights_multicast_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1.,  x, ones_C_.gpu_data(),0., temp_NC.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., temp_NC.gpu_data(), ones_HW_.gpu_data(), 0., y);
}


// y[c] = sum x(.,c,...)
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::compute_sum_per_channel_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  // assume that x.shape(1)==C
  Blob<Dtype> templ_NC;
  vector<int> templ_size;
  templ_size.push_back(N*C);
  templ_NC.Reshape(templ_size);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, 1., x, ones_HW_.gpu_data(),
      0., templ_NC.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, N, C, 1., templ_NC.gpu_data(),
      ones_N_.gpu_data(), 0., y);
}


// y[c] = sum x(.,c,...)
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::compute_sum_per_sample_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  // assume that x.shape(1)==C
  Blob<Dtype> templ_NC;
  vector<int> templ_size;
  templ_size.push_back(N*C);
  templ_NC.Reshape(templ_size);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, 1., x, ones_HW_.gpu_data(),
      0., templ_NC.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, N, C, 1., templ_NC.gpu_data(),
      ones_C_.gpu_data(), 0., y);
}

// y[c] = mean x(.,c,...)
template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::compute_mean_per_channel_gpu(int N, int C, int S,
    const Dtype *x, const Dtype *w, Dtype *y ) {
  	Blob<Dtype> templ_C;
  	vector<int> templ_size;
  	templ_size.push_back(C);
  	templ_C.Reshape(templ_size);
        Blob<Dtype> app_;
  	vector<int> app_size;
  	app_size.push_back(N*C*S);
  	app_.Reshape(app_size);
	weights_multicast_gpu(N,C,S,w,app_.mutable_gpu_data());			// W \in NCHW
	compute_sum_per_channel_gpu(N,C,S,app_.gpu_data(),templ_C.mutable_gpu_data()); 	// sum W per channel
	caffe_gpu_mul(N*C*S,app_.gpu_data(), x,app_.mutable_gpu_data()); 				// W*X eltwise
	
	caffe_gpu_powx(C, templ_C.gpu_data(), Dtype(-1.0), templ_C.mutable_gpu_data());	// 1/sumW
	
	compute_sum_per_channel_gpu(N,C,S,app_.gpu_data(),y);				//sumWX
	caffe_gpu_mul(C, y, templ_C.gpu_data(), y);						//scale sumWX/sumW
}






template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weights = bottom[1]->gpu_data();

  if (use_global_stats_) {
    // use global mean/variance
    caffe_copy(C, this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_copy(C, this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    compute_mean_per_channel_gpu(N, C, S, bottom_data, weights,
        mean_.mutable_gpu_data());
  }

  //  Y = X- EX
  if (bottom[0] != top[0]) {
    caffe_copy(top_size, bottom_data, top_data);
  }
  multicast_gpu(N, C, S, mean_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_axpby(top_size, Dtype(-1.), temp_.gpu_data(),
      Dtype(1.), top_data);

  if (!use_global_stats_) {
     //  compute variance E (X-EX)^2
    caffe_gpu_powx(top_size, top_data, Dtype(2.), temp_.mutable_gpu_data());
    compute_mean_per_channel_gpu(N, C, S, temp_.gpu_data(), weights,
        variance_.mutable_gpu_data());
    // int m = N*S;    // m = N*H*W
    // Dtype bias_corr = m > 1 ? Dtype(m)/(m-1) : 1;
    // bias_corr = 1.;
    // caffe_gpu_scale(C, bias_corr, variance_.gpu_data(),
    //    variance_.mutable_gpu_data());

    // clip variance
    if ((this->phase_ == TRAIN) && (iter_ <= MULTIMODAL_BN_VARIANCE_CLIP_START))
      iter_++;
    if (iter_ > MULTIMODAL_BN_VARIANCE_CLIP_START) {
      // clip from above
      // temp_C_[c] = average + gobal_var[c]
      Dtype y;
      caffe_gpu_asum(C, this->blobs_[1]->gpu_data(), &y);
      caffe_gpu_scale(C, Dtype(y/C), ones_C_.gpu_data(),
          temp_C_.mutable_gpu_data());
      caffe_gpu_axpby(C, Dtype(1.0), this->blobs_[1]->gpu_data(),
          Dtype(1.0), temp_C_.mutable_gpu_data());
      caffe_gpu_eltwise_min(C,
          Dtype(MULTIMODAL_BN_VARIANCE_CLIP_CONST), temp_C_.gpu_data(),
          Dtype(1.0), variance_.mutable_gpu_data());
      // clip from below
      caffe_gpu_eltwise_max(C,
          Dtype((1.)/MULTIMODAL_BN_VARIANCE_CLIP_CONST), this->blobs_[1]->gpu_data(),
          Dtype(1.0), variance_.mutable_gpu_data());
    }
    //  update global mean and variance
    if (iter_ > 1) {
      caffe_gpu_axpby(C,
        Dtype(1. - moving_average_fraction_), mean_.gpu_data(),
        Dtype(moving_average_fraction_), this->blobs_[0]->mutable_gpu_data());
      caffe_gpu_axpby(C,
        Dtype((1.- moving_average_fraction_)), variance_.gpu_data(),
        Dtype(moving_average_fraction_), this->blobs_[1]->mutable_gpu_data());
    } else {
      caffe_copy(C, mean_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
      caffe_copy(C, variance_.gpu_data(), this->blobs_[1]->mutable_gpu_data());
    }
  }
  //  inv_var = (eps + variance)^(-0.5)
  caffe_gpu_add_scalar(C, eps_, variance_.mutable_gpu_data());
  caffe_gpu_powx(C, variance_.gpu_data(), Dtype(-0.5),
      inv_variance_.mutable_gpu_data());

  //  X_norm = (X-EX) * inv_var
  multicast_gpu(N, C, S, inv_variance_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, top_data, temp_.gpu_data(), top_data);

  // copy x_norm for backward
  caffe_copy(top_size, top_data, x_norm_.mutable_gpu_data());
  weights_multicast_gpu(N,C,S,weights,temp_.mutable_gpu_data());

  caffe_gpu_mul(top_size, top_data, temp_.gpu_data(), top_data); // (X-E[X])/std * w



  

}

template <typename Dtype>
void MultiModalBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();
  const Dtype* weights = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();

  // --  STAGE 1: backprop dE/d(x_norm) = dE/dY .* w ------------
  weights_multicast_gpu(N, C, S, weights, temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, top_diff, temp_.gpu_data(), x_norm_.mutable_gpu_diff());

  // --  STAGE : backprop dE/dY --> dE/dX --------------------------

  // ATTENTION: from now on we will use notation Y:= X_norm
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =  (dE/dY - w*sum(dE/dY)/sum(w) - w*sum(dE/dY .* Y) .* Y)/sum(w)
  //             ./ sqrt(var(X) + eps)
  // where
  // .* and ./ are element-wise product and division,
  // mean, var, sum are computed along all dimensions except the channels.

  const Dtype* top_data = x_norm_.gpu_data();
  top_diff = x_norm_.gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* weights_diff = bottom[1]->mutable_gpu_diff();

  // temp = sum(dE/dY .* Y)
  caffe_gpu_mul(top_size, top_diff, top_data, temp_.mutable_gpu_diff());
  compute_sum_per_channel_gpu(N, C, S, temp_.gpu_diff(),
      temp_C_.mutable_gpu_diff());
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());

  // bottom = sum(dE/dY .* Y) .* Y
  caffe_gpu_mul(top_size, temp_.gpu_diff(), top_data, bottom_diff);

  // temp = sum(dE/dY)
  compute_sum_per_channel_gpu(N, C, S, top_diff, temp_C_.mutable_gpu_diff());
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());

  // bottom = (sum(dE/dY) + sum(dE/dY .* Y) .* Y)*w
  caffe_gpu_add(top_size, temp_.gpu_diff(), bottom_diff, bottom_diff);

  weights_multicast_gpu(N, C, S, weights, temp_.mutable_gpu_data());
  compute_sum_per_channel_gpu(N,C,S,temp_.gpu_data(),temp_C_.mutable_gpu_data()); 
  caffe_gpu_powx(C, temp_C_.gpu_data(), Dtype(-1.0),
      temp_C_.mutable_gpu_data()); // std = (var+eps)^(-0.5)

  caffe_gpu_mul(top_size, bottom_diff, temp_.gpu_data(), bottom_diff); 		//diff*w
  multicast_gpu(N, C, S, temp_C_.gpu_data(), temp_.mutable_gpu_data());         
  caffe_gpu_mul(top_size, bottom_diff, temp_.gpu_data(), bottom_diff);		//diff*(1/w)

  // bottom = dE/dY - (sum(dE/dY)-msum(dE/dY \cdot Y) \cdot Y)*w/sum(w)
  caffe_gpu_axpby(top_size, Dtype(1.), top_diff, Dtype(-1.), bottom_diff);

  // dE/dX = dE/dX ./ sqrt(var(X) + eps)
  multicast_gpu(N, C, S, inv_variance_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, bottom_diff, temp_.gpu_data(), bottom_diff); 

 // STAGE 3 Propagate error to weights
  //
  // We will compute (considering this mode as M):
  // dE/dw = dE/dM * Y -Y/sw.*sum(dE/dY) + 1/(2*sw).*(1-Y^2).*sum(dE/dY.*Y)
  //

  // temp = mean(dE/dY .* Y)
 // 1
  caffe_gpu_mul(top_size, top[0]->gpu_diff(), top_data, temp_.mutable_gpu_diff());
  compute_sum_per_sample_gpu(N, C, S, temp_.gpu_diff(), weights_diff);

  // temp = mean(dE/dY .* Y)
  caffe_gpu_mul(top_size, top_diff, top_data, temp_.mutable_gpu_diff()); 				// dE/dY .* Y
  compute_sum_per_channel_gpu(N, C, S, temp_.gpu_diff(), temp_C_.mutable_gpu_diff());   	// sum(dE/dY .* Y)
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());				// sum repeated

  // bottom_w = -0.5*sum(dE/dY .* Y) .* Y^2

  caffe_gpu_mul(top_size, top_data, top_data, temp2_.mutable_gpu_diff());				// Y^2
  caffe_gpu_add_scalar(top_size, Dtype(-1.0), temp2_.mutable_gpu_diff());				// -(1-Y^2)
  caffe_gpu_mul(top_size, temp_.gpu_diff(), temp2_.gpu_diff(), temp2_.mutable_gpu_diff());		// temp2 = -(1-Y^2).*sum(dE/dY.*Y)

  

  compute_sum_per_channel_gpu(N, C, S, top_diff, temp_C_.mutable_gpu_diff()); 			// temp = sum(dE/dY)
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());				// Repeated
  

  caffe_gpu_mul(top_size, temp_.gpu_diff(), top_data, temp_.mutable_gpu_diff());			// temp = Y.*sum(dE/dY)
  caffe_gpu_axpby(top_size, Dtype(-1.), temp_.gpu_diff(), Dtype(-0.5), temp2_.mutable_gpu_diff()); // -Y/sw.*sum(dE/dY) + 1/(2*sw).*(1-Y^2).*sum(dE/dY.*Y)

 
  weights_multicast_gpu(N, C, S, weights, temp_.mutable_gpu_data());
  compute_sum_per_channel_gpu(N,C,S,temp_.gpu_data(),temp_C_.mutable_gpu_data()); 
  caffe_gpu_powx(C, temp_C_.gpu_data(), Dtype(-1.0),
      temp_C_.mutable_gpu_data()); // std = (var+eps)^(-0.5)
  multicast_gpu(N, C, S, temp_C_.gpu_data(), temp_.mutable_gpu_data());         
  caffe_gpu_mul(top_size, temp2_.gpu_diff(), temp_.gpu_data(), temp2_.mutable_gpu_diff());



  compute_sum_per_sample_gpu(N, C, S, temp2_.gpu_diff(), temp_N_.mutable_gpu_diff());
  caffe_gpu_add(N, temp_N_.gpu_diff(), weights_diff, weights_diff);

   int i=1;
if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_gpu_diff());
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(MultiModalBatchNormLayer);

template void MultiModalBatchNormLayer<float>::multicast_gpu(int N, int C,
    int S, const float *x, float *y);
template void MultiModalBatchNormLayer<float>::weights_multicast_gpu(int N, int C,
    int S, const float *x, float *y);
template void MultiModalBatchNormLayer<float>::compute_sum_per_channel_gpu(int N, int C,
    int S, const float *x, float *y);
template void MultiModalBatchNormLayer<float>::compute_mean_per_channel_gpu(int N, int C,
    int S, const float *x, const float *w, float *y);
template void MultiModalBatchNormLayer<float>::compute_sum_per_sample_gpu(int N, int C,
    int S, const float *x, float *y);

template void MultiModalBatchNormLayer<double>::multicast_gpu(int N, int C,
    int S, const double *x, double *y);
template void MultiModalBatchNormLayer<double>::weights_multicast_gpu(int N, int C,
    int S, const double *x, double *y);
template void MultiModalBatchNormLayer<double>::compute_sum_per_channel_gpu(int N, int C,
    int S, const double *x, double *y);
template void MultiModalBatchNormLayer<double>::compute_mean_per_channel_gpu(int N, int C,
    int S, const double *x, const double *w, double *y);
template void MultiModalBatchNormLayer<double>::compute_sum_per_sample_gpu(int N, int C,
    int S, const double *x, double *y);

}  // namespace caffe
