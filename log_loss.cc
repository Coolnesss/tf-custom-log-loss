#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("LogLoss")
    .Input("pred: float32")
    .Input("real: float32")
    .Output("loss: float32");


class LogLossOp : public OpKernel {
 public:
  explicit LogLossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    
    // Grab the prediction tensor
    const Tensor& pred_tensor = context->input(0);

    // Grab the y tensor
    const Tensor& y_tensor = context->input(1);

    //std::cout << "SHAPE";// << pred_tensor.shape() << y_tensor.shape() << std::endl;
    //auto input = input_tensor.flat<float64>();

    // Create an output tensor
    Tensor* loss_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, pred_tensor.shape(),
                                                    &loss_tensor));
    //auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    //const int N = input.size();
    //for (int i = 1; i < N; i++) {
    //  output_flat(i) = 0;
    //}

    // Preserve the first input value if possible.
    //if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("LogLoss").Device(DEVICE_CPU), LogLossOp);
