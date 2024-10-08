//
//  CutlassGemmTuneCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2023/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef ENABLE_CUDA_TUNE_PARAM

#ifndef CutlassGemmTuneCommonExecution_hpp
#define CutlassGemmTuneCommonExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "execution/cutlass_common/tune/CutlassGemmTune.hpp"

namespace MNN {
namespace CUDA {

class CutlassGemmTuneCommonExecution : public Execution {
public:
    CutlassGemmTuneCommonExecution(Backend* backend) : Execution(backend) {};
    virtual ~CutlassGemmTuneCommonExecution() = default;

    void setGemmBatchedTensorCoreFloat16Argments(const GemmParamInfo* params);
    void runGemmBatchedTensorCoreFloat16Infer(const GemmParamInfo* params);
    void setGemmTensorCoreFloat16Argments(const GemmParamInfo* params);
    void runGemmTensorCoreFloat16Infer(const GemmParamInfo* params);
protected:
    GemmParamInfo mInfo;

    // GemmBatched
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32  mGemmBatchedF16F16TensorAlign8RC_64x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64  mGemmBatchedF16F16TensorAlign8RC_64x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32 mGemmBatchedF16F16TensorAlign8RC_64x128x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32 mGemmBatchedF16F16TensorAlign8RC_128x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64 mGemmBatchedF16F16TensorAlign8RC_128x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32 mGemmBatchedF16F16TensorAlign8RC_256x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32 mGemmBatchedF16F16TensorAlign8RC_128x128x32;

    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x32    mGemmBatchedF16F16TensorAlign1RC_64x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x64x64    mGemmBatchedF16F16TensorAlign1RC_64x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_64x128x32   mGemmBatchedF16F16TensorAlign1RC_64x128x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x32   mGemmBatchedF16F16TensorAlign1RC_128x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x64x64   mGemmBatchedF16F16TensorAlign1RC_128x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_256x64x32   mGemmBatchedF16F16TensorAlign1RC_256x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm80_128x128x32   mGemmBatchedF16F16TensorAlign1RC_128x128x32;

    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x32  mGemmBatchedF16F16TensorAlign8RR_64x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x64x64  mGemmBatchedF16F16TensorAlign8RR_64x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_64x128x32 mGemmBatchedF16F16TensorAlign8RR_64x128x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x32 mGemmBatchedF16F16TensorAlign8RR_128x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x64x64 mGemmBatchedF16F16TensorAlign8RR_128x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_256x64x32 mGemmBatchedF16F16TensorAlign8RR_256x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm80_128x128x32 mGemmBatchedF16F16TensorAlign8RR_128x128x32;

    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x32    mGemmBatchedF16F16TensorAlign1RR_64x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x64x64    mGemmBatchedF16F16TensorAlign1RR_64x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_64x128x32   mGemmBatchedF16F16TensorAlign1RR_64x128x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x32   mGemmBatchedF16F16TensorAlign1RR_128x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x64x64   mGemmBatchedF16F16TensorAlign1RR_128x64x64;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_256x64x32   mGemmBatchedF16F16TensorAlign1RR_256x64x32;
    GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm80_128x128x32   mGemmBatchedF16F16TensorAlign1RR_128x128x32;

    // // Gemm Linear
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F16TensorLnAlign8RC_64x64x32;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F16TensorLnAlign8RC_64x64x64;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F16TensorLnAlign8RC_64x128x32;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F16TensorLnAlign8RC_128x64x32;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F16TensorLnAlign8RC_128x64x64;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F16TensorLnAlign8RC_256x64x32;
    GemmTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F16TensorLnAlign8RC_128x128x32;

    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F32TensorLnAlign8RC_64x64x32;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F32TensorLnAlign8RC_64x64x64;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F32TensorLnAlign8RC_64x128x32;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F32TensorLnAlign8RC_128x64x32;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F32TensorLnAlign8RC_128x64x64;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F32TensorLnAlign8RC_256x64x32;
    GemmTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F32TensorLnAlign8RC_128x128x32;

    // // Gemm Relu
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F16TensorReluAlign8RC_64x64x32;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F16TensorReluAlign8RC_64x64x64;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F16TensorReluAlign8RC_64x128x32;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F16TensorReluAlign8RC_128x64x32;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F16TensorReluAlign8RC_128x64x64;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F16TensorReluAlign8RC_256x64x32;
    GemmTensor_F16_F16_Relu_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F16TensorReluAlign8RC_128x128x32;

    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F32TensorReluAlign8RC_64x64x32;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F32TensorReluAlign8RC_64x64x64;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F32TensorReluAlign8RC_64x128x32;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F32TensorReluAlign8RC_128x64x32;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F32TensorReluAlign8RC_128x64x64;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F32TensorReluAlign8RC_256x64x32;
    GemmTensor_F16_F32_Relu_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F32TensorReluAlign8RC_128x128x32;

    // // Gemm Relu6
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F16TensorRelu6Align8RC_64x64x32;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F16TensorRelu6Align8RC_64x64x64;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F16TensorRelu6Align8RC_64x128x32;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F16TensorRelu6Align8RC_128x64x32;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F16TensorRelu6Align8RC_128x64x64;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F16TensorRelu6Align8RC_256x64x32;
    GemmTensor_F16_F16_Relu6_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F16TensorRelu6Align8RC_128x128x32;

    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x32  mGemmF16F32TensorRelu6Align8RC_64x64x32;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x64x64  mGemmF16F32TensorRelu6Align8RC_64x64x64;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_64x128x32 mGemmF16F32TensorRelu6Align8RC_64x128x32;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x32 mGemmF16F32TensorRelu6Align8RC_128x64x32;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x64x64 mGemmF16F32TensorRelu6Align8RC_128x64x64;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_256x64x32 mGemmF16F32TensorRelu6Align8RC_256x64x32;
    GemmTensor_F16_F32_Relu6_AlignTensor_Row_Column_Sm80_128x128x32 mGemmF16F32TensorRelu6Align8RC_128x128x32;

    int mGpuComputeCap = 75;
    int mActivationType = 0;
    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;
    bool mBf16Infer = false;
    int mPrecisonLevel;
    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
};

} // namespace CUDA
} // namespace MNN

#endif /* CutlassGemmTuneCommonExecution */
#endif