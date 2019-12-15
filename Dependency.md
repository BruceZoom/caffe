# ResNet18
- **conv_layer.cu**
  - ConvolutionLayer::Forward_gpu
    - BaseConvolutionLayer::forward_gpu_gemm
      - ```c
        conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
      - ```c
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_,
            (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
            (Dtype)0., output + output_offset_ * g);
    - BaseConvolutionLayer::forward_gpu_bias
      - ```
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
            (Dtype)1., output);
  - ConvolutionLayer::Backward_gpu
    - BaseConvolutionLayer::backward_gpu_bias
      - ```c
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
            input, bias_multiplier_.gpu_data(), 1., bias);
    - BaseConvolutionLayer::weight_gpu_gemm
      - ```c
        conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
      - ```c
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
            kernel_dim_, conv_out_spatial_dim_,
            (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
            (Dtype)1., weights + weight_offset_ * g);
    - BaseConvolutionLayer::backward_gpu_gemm
      - ```c
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
            conv_out_spatial_dim_, conv_out_channels_ / group_,
            (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
            (Dtype)0., col_buff + col_offset_ * g);
      - ```c
        conv_col2im_gpu(col_buff, input);
- **scale_layer.cu**
  - ScaleLayer::Forward_gpu
    - ```c
      ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
        top_data);
    - ```c
      ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
  - ScaleLayer::Backward_gpu
    - ```c
      bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
    - ```c
      caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    - ```c
       caffe_gpu_dot(inner_dim_, product, sum_mult, &result);
       ```
       caffe_gpu_dot * 4 in total
    - ```c
      caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      ```
      caffe_gpu_gemv * 2 in total
    - ```c
      ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, scale_dim_, inner_dim_, bottom_diff);
      ```
- **inner_product_layer.cu**
  - InnerProductLayer::Forward_gpu
    - ```c
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                           weight, bottom_data, (Dtype)0., top_data);
    - ```
      caffe_gpu_gemm<Dtype>(CblasNoTrans,-
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, (Dtype)1.,
                            bottom_data, weight, (Dtype)0., top_data);
      ```
      caffe_gpu_gemm * 2 in total
    - ```c
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  - InnerProductLayer::Backward_gpu
    - ```c
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            K_, N_, M_,
            (Dtype)1., bottom_data, top_diff,
            (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      ```
      caffe_gpu_gemm * 4 in total
    - ```c
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.gpu_data(), (Dtype)1.,
            this->blobs_[1]->mutable_gpu_diff());
- **pooling_layer.cu**
  - PoolingLayer::Forward_gpu
    - ```c
      void MaxPoolForward(const int nthreads,
            const Dtype* const bottom_data, const int num, const int channels,
            const int height, const int width, const int pooled_height,
            const int pooled_width, const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w, const int pad_h, const int pad_w,
            Dtype* const top_data, int* mask, Dtype* top_mask)
    - ```c
      void AvePoolForward(const int nthreads,
            const Dtype* const bottom_data, const int num, const int channels,
            const int height, const int width, const int pooled_height,
            const int pooled_width, const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w, const int pad_h, const int pad_w,
            Dtype* const top_data)
    - ```c
      void StoPoolForwardTrain(const int nthreads,
            const Dtype* const bottom_data,
            const int num, const int channels, const int height,
            const int width, const int pooled_height, const int pooled_width,
            const int kernel_h, const int kernel_w, const int stride_h,
            const int stride_w, Dtype* const rand_idx, Dtype* const top_data)
    - ```c
      void StoPoolForwardTrain(const int nthreads,
            const Dtype* const bottom_data,
            const int num, const int channels, const int height,
            const int width, const int pooled_height, const int pooled_width,
            const int kernel_h, const int kernel_w, const int stride_h,
            const int stride_w, Dtype* const rand_idx, Dtype* const top_data)
    - ```c
      void StoPoolForwardTest(const int nthreads,
            const Dtype* const bottom_data,
            const int num, const int channels, const int height,
            const int width, const int pooled_height, const int pooled_width,
            const int kernel_h, const int kernel_w, const int stride_h,
            const int stride_w, Dtype* const top_data)
  - PoolingLayer::Backward_gpu
    - ```c
      MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
            const int* const mask, const Dtype* const top_mask, const int num,
            const int channels, const int height, const int width,
            const int pooled_height, const int pooled_width, const int kernel_h,
            const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
            const int pad_w, Dtype* const bottom_diff)
    - ```c
      void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
            const int num, const int channels, const int height,
            const int width, const int pooled_height, const int pooled_width,
            const int kernel_h, const int kernel_w, const int stride_h,
            const int stride_w, const int pad_h, const int pad_w, Dtype* const bottom_diff)
    - ```c
      void StoPoolBackward(const int nthreads,
            const Dtype* const rand_idx, const Dtype* const top_diff,
            const int num, const int channels, const int height,
            const int width, const int pooled_height, const int pooled_width,
            const int kernel_h, const int kernel_w, const int stride_h,
            const int stride_w, Dtype* const bottom_diff)
- **eltwise_layer.cu**
- **softmax_layer.cu**
- **relu_layer.cu**
- **batch_norm_layer.cu**

## Inception
- **concat_layer.cu**
- *constant?*
- *xavier?*