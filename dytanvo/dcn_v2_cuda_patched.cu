#include <vector>
#include "cuda/dcn_v2_im2col_cuda.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int kernel_h, const int kernel_w,
                    const int stride_h, const int stride_w,
                    const int pad_h,    const int pad_w,
                    const int dilation_h, const int dilation_w,
                    const int deformable_group)
{
    using scalar_t = float;
    AT_ASSERTM(input.is_cuda(),  "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(),   "bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(),   "mask must be a CUDA tensor");

    const int batch        = input.size(0);
    const int channels     = input.size(1);
    const int height       = input.size(2);
    const int width        = input.size(3);
    const int channels_out = weight.size(0);
    const int kernel_h_    = weight.size(2);
    const int kernel_w_    = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "kernel shape mismatch");
    AT_ASSERTM(weight.size(1) == channels,
               "channel mismatch");

    const int height_out = (height + 2*pad_h - (dilation_h*(kernel_h-1)+1)) / stride_h + 1;
    const int width_out  = (width  + 2*pad_w - (dilation_w*(kernel_w-1)+1)) / stride_w + 1;

    // weight_2d: (channels_out) x (channels*kH*kW)
    auto weight_2d = weight.view({channels_out, channels * kernel_h * kernel_w});
    // output: (batch, channels_out, h_out, w_out)
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());
    // columns: (channels*kH*kW) x (h_out*w_out)
    auto columns = at::empty({channels * kernel_h * kernel_w, height_out * width_out}, input.options());

    for (int b = 0; b < batch; b++)
    {
        auto input_n  = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n   = mask.select(0, b);

        modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                         input_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<scalar_t>());

        // output_n (channels_out x h_out*w_out) = weight_2d @ columns + bias
        // используем at::mm — чистый PyTorch CUDA GEMM без cuBLAS напрямую
        auto output_n = output.select(0, b)
                              .view({channels_out, height_out * width_out});
        at::mm_out(output_n, weight_2d, columns);
        output_n.add_(bias.unsqueeze(1));
    }

    return output;
}

std::vector<at::Tensor> dcn_v2_cuda_backward(const at::Tensor &input,
                                              const at::Tensor &weight,
                                              const at::Tensor &bias,
                                              const at::Tensor &offset,
                                              const at::Tensor &mask,
                                              const at::Tensor &grad_output,
                                              int kernel_h, int kernel_w,
                                              int stride_h, int stride_w,
                                              int pad_h,    int pad_w,
                                              int dilation_h, int dilation_w,
                                              int deformable_group)
{
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int batch        = input.size(0);
    const int channels     = input.size(1);
    const int height       = input.size(2);
    const int width        = input.size(3);
    const int channels_out = weight.size(0);

    const int height_out = (height + 2*pad_h - (dilation_h*(kernel_h-1)+1)) / stride_h + 1;
    const int width_out  = (width  + 2*pad_w - (dilation_w*(kernel_w-1)+1)) / stride_w + 1;

    auto weight_2d = weight.view({channels_out, channels * kernel_h * kernel_w});
    auto columns   = at::empty({channels * kernel_h * kernel_w, height_out * width_out}, input.options());

    auto grad_input  = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias   = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask   = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n       = input.select(0, b);
        auto offset_n      = offset.select(0, b);
        auto mask_n        = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b)
                                        .view({channels_out, height_out * width_out});
        auto grad_input_n  = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n   = grad_mask.select(0, b);

        // columns = weight_2d^T @ grad_output_n
        columns = at::mm(weight_2d.t(), grad_output_n);

        modulated_deformable_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                               columns.data_ptr<scalar_t>(),
                                               input_n.data_ptr<scalar_t>(),
                                               offset_n.data_ptr<scalar_t>(),
                                               mask_n.data_ptr<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<scalar_t>(),
                                               grad_mask_n.data_ptr<scalar_t>());

        modulated_deformable_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                                         columns.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<scalar_t>());

        modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                         input_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<scalar_t>());

        // grad_weight += grad_output_n @ columns^T
        grad_weight.view({channels_out, -1}).add_(at::mm(grad_output_n, columns.t()));

        // grad_bias += sum over spatial
        grad_bias.add_(grad_output_n.sum(1));
    }

    return {grad_input, grad_offset, grad_mask, grad_weight, grad_bias};
}
