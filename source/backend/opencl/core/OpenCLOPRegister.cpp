// This file is generated by Shell for ops register
#ifndef MNN_OPENCL_SEP_BUILD
namespace MNN {
namespace OpenCL {
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern void ___OpenCLInterp3DBufCreator__OpType_Interp3D__BUFFER__();
extern void ___OpenCLReductionBufCreator__OpType_Reduction__BUFFER__();
extern void ___OpenCLArgMaxBufCreator__OpType_ArgMax__BUFFER__();
extern void ___OpenCLArgMaxBufCreator__OpType_ArgMin__BUFFER__();
extern void ___OpenCLMatMulBufCreator__OpType_MatMul__BUFFER__();
extern void ___OpenCLRasterBufCreator__OpType_Raster__BUFFER__();
extern void ___OpenCLLayerNormBufCreator__OpType_LayerNorm__BUFFER__();
extern void ___OpenCLDepthwiseConvolutionBufCreator__OpType_ConvolutionDepthwise__BUFFER__();
extern void ___OpenCLInterpBufCreator__OpType_Interp__BUFFER__();
extern void ___OpenCLBinaryBufCreator__OpType_Eltwise__BUFFER__();
extern void ___OpenCLBinaryBufCreator__OpType_BinaryOp__BUFFER__();
extern void ___OpenCLConvolutionBufCreator__OpType_Convolution__BUFFER__();
extern void ___OpenCLSelectBufCreator__OpType_Select__BUFFER__();
extern void ___OpenCLPoolBufCreator__OpType_Pooling__BUFFER__();
extern void ___OpenCLDeconvolutionBufCreator__OpType_Deconvolution__BUFFER__();
extern void ___OpenCLCastBufCreator__OpType_Cast__BUFFER__();
extern void ___OpenCLReluBufCreator__OpType_ReLU__BUFFER__();
extern void ___OpenCLReluBufCreator__OpType_PReLU__BUFFER__();
extern void ___OpenCLReluBufCreator__OpType_ReLU6__BUFFER__();
extern void ___OpenCLSoftmaxBufCreator__OpType_Softmax__BUFFER__();
extern void ___OpenCLLoopBufCreator__OpType_While__BUFFER__();
extern void ___OpenCLRangeBufCreator__OpType_Range__BUFFER__();
extern void ___OpenCLUnaryBufCreator__OpType_UnaryOp__BUFFER__();
extern void ___OpenCLUnaryBufCreator__OpType_Sigmoid__BUFFER__();
extern void ___OpenCLUnaryBufCreator__OpType_TanH__BUFFER__();
extern void ___OpenCLGridSampleBufCreator__OpType_GridSample__BUFFER__();
extern void ___OpenCLScaleBufCreator__OpType_Scale__BUFFER__();
#endif
extern void ___OpenCLDepthwiseConvolutionCreator__OpType_ConvolutionDepthwise__IMAGE__();
extern void ___OpenCLMatMulCreator__OpType_MatMul__IMAGE__();
extern void ___OpenCLUnaryCreator__OpType_UnaryOp__IMAGE__();
extern void ___OpenCLUnaryCreator__OpType_Sigmoid__IMAGE__();
extern void ___OpenCLUnaryCreator__OpType_TanH__IMAGE__();
extern void ___OpenCLScaleCreator__OpType_Scale__IMAGE__();
extern void ___OpenCLSoftmaxCreator__OpType_Softmax__IMAGE__();
extern void ___OpenCLEltwiseCreator__OpType_Eltwise__IMAGE__();
extern void ___OpenCLEltwiseCreator__OpType_BinaryOp__IMAGE__();
extern void ___OpenCLRangeCreator__OpType_Range__IMAGE__();
extern void ___OpenCLRasterCreator__OpType_Raster__IMAGE__();
extern void ___OpenCLFuseCreator__OpType_Extra__IMAGE__();
extern void ___OpenCLLoopCreator__OpType_While__IMAGE__();
extern void ___OpenCLTrainableParamCreator__OpType_TrainableParam__IMAGE__();
extern void ___OpenCLReluCreator__OpType_ReLU__IMAGE__();
extern void ___OpenCLReluCreator__OpType_PReLU__IMAGE__();
extern void ___OpenCLReluCreator__OpType_ReLU6__IMAGE__();
extern void ___OpenCLConvolutionCreator__OpType_Convolution__IMAGE__();
extern void ___OpenCLLayerNormCreator__OpType_LayerNorm__IMAGE__();
extern void ___OpenCLReductionCreator__OpType_Reduction__IMAGE__();
extern void ___OpenCLRoiPoolingCreator__OpType_ROIPooling__IMAGE__();
extern void ___OpenCLPoolCreator__OpType_Pooling__IMAGE__();
extern void ___OpenCLSelectCreator__OpType_Select__IMAGE__();
extern void ___OpenCLDeconvolutionCreator__OpType_Deconvolution__IMAGE__();
extern void ___OpenCLDepthwiseDeconvolutionCreator__OpType_DeconvolutionDepthwise__IMAGE__();
extern void ___OpenCLInterp3DCreator__OpType_Interp3D__IMAGE__();
extern void ___OpenCLCastCreator__OpType_Cast__IMAGE__();
extern void ___OpenCLInterpCreator__OpType_Interp__IMAGE__();
extern void ___OpenCLGridSampleCreator__OpType_GridSample__IMAGE__();

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
extern void ___OpenCLSelfAttentionBufCreator__OpType_FmhaV2__BUFFER__();
extern void ___OpenCLSplitGeluBufCreator__OpType_SplitGeLU__BUFFER__();
extern void ___OpenCLGroupNormBufCreator__OpType_GroupNorm__BUFFER__();
extern void ___OpenCLAttentionBufCreator__OpType_Attention__BUFFER__();
#endif
void registerOpenCLOps() {
#ifndef MNN_OPENCL_BUFFER_CLOSED
___OpenCLInterp3DBufCreator__OpType_Interp3D__BUFFER__();
___OpenCLReductionBufCreator__OpType_Reduction__BUFFER__();
___OpenCLArgMaxBufCreator__OpType_ArgMax__BUFFER__();
___OpenCLArgMaxBufCreator__OpType_ArgMin__BUFFER__();
___OpenCLMatMulBufCreator__OpType_MatMul__BUFFER__();
___OpenCLRasterBufCreator__OpType_Raster__BUFFER__();
___OpenCLLayerNormBufCreator__OpType_LayerNorm__BUFFER__();
___OpenCLDepthwiseConvolutionBufCreator__OpType_ConvolutionDepthwise__BUFFER__();
___OpenCLInterpBufCreator__OpType_Interp__BUFFER__();
___OpenCLBinaryBufCreator__OpType_Eltwise__BUFFER__();
___OpenCLBinaryBufCreator__OpType_BinaryOp__BUFFER__();
___OpenCLConvolutionBufCreator__OpType_Convolution__BUFFER__();
___OpenCLSelectBufCreator__OpType_Select__BUFFER__();
___OpenCLPoolBufCreator__OpType_Pooling__BUFFER__();
___OpenCLDeconvolutionBufCreator__OpType_Deconvolution__BUFFER__();
___OpenCLCastBufCreator__OpType_Cast__BUFFER__();
___OpenCLReluBufCreator__OpType_ReLU__BUFFER__();
___OpenCLReluBufCreator__OpType_PReLU__BUFFER__();
___OpenCLReluBufCreator__OpType_ReLU6__BUFFER__();
___OpenCLSoftmaxBufCreator__OpType_Softmax__BUFFER__();
___OpenCLLoopBufCreator__OpType_While__BUFFER__();
___OpenCLRangeBufCreator__OpType_Range__BUFFER__();
___OpenCLUnaryBufCreator__OpType_UnaryOp__BUFFER__();
___OpenCLUnaryBufCreator__OpType_Sigmoid__BUFFER__();
___OpenCLUnaryBufCreator__OpType_TanH__BUFFER__();
___OpenCLGridSampleBufCreator__OpType_GridSample__BUFFER__();
___OpenCLScaleBufCreator__OpType_Scale__BUFFER__();
#endif
___OpenCLDepthwiseConvolutionCreator__OpType_ConvolutionDepthwise__IMAGE__();
___OpenCLMatMulCreator__OpType_MatMul__IMAGE__();
___OpenCLUnaryCreator__OpType_UnaryOp__IMAGE__();
___OpenCLUnaryCreator__OpType_Sigmoid__IMAGE__();
___OpenCLUnaryCreator__OpType_TanH__IMAGE__();
___OpenCLScaleCreator__OpType_Scale__IMAGE__();
___OpenCLSoftmaxCreator__OpType_Softmax__IMAGE__();
___OpenCLEltwiseCreator__OpType_Eltwise__IMAGE__();
___OpenCLEltwiseCreator__OpType_BinaryOp__IMAGE__();
___OpenCLRangeCreator__OpType_Range__IMAGE__();
___OpenCLRasterCreator__OpType_Raster__IMAGE__();
___OpenCLFuseCreator__OpType_Extra__IMAGE__();
___OpenCLLoopCreator__OpType_While__IMAGE__();
___OpenCLTrainableParamCreator__OpType_TrainableParam__IMAGE__();
___OpenCLReluCreator__OpType_ReLU__IMAGE__();
___OpenCLReluCreator__OpType_PReLU__IMAGE__();
___OpenCLReluCreator__OpType_ReLU6__IMAGE__();
___OpenCLConvolutionCreator__OpType_Convolution__IMAGE__();
___OpenCLLayerNormCreator__OpType_LayerNorm__IMAGE__();
___OpenCLReductionCreator__OpType_Reduction__IMAGE__();
___OpenCLRoiPoolingCreator__OpType_ROIPooling__IMAGE__();
___OpenCLPoolCreator__OpType_Pooling__IMAGE__();
___OpenCLSelectCreator__OpType_Select__IMAGE__();
___OpenCLDeconvolutionCreator__OpType_Deconvolution__IMAGE__();
___OpenCLDepthwiseDeconvolutionCreator__OpType_DeconvolutionDepthwise__IMAGE__();
___OpenCLInterp3DCreator__OpType_Interp3D__IMAGE__();
___OpenCLCastCreator__OpType_Cast__IMAGE__();
___OpenCLInterpCreator__OpType_Interp__IMAGE__();
___OpenCLGridSampleCreator__OpType_GridSample__IMAGE__();

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
___OpenCLSelfAttentionBufCreator__OpType_FmhaV2__BUFFER__();
___OpenCLSplitGeluBufCreator__OpType_SplitGeLU__BUFFER__();
___OpenCLGroupNormBufCreator__OpType_GroupNorm__BUFFER__();
___OpenCLAttentionBufCreator__OpType_Attention__BUFFER__();
#endif
}
}
}
#endif
