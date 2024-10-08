//
//  TFliteBatchToSpace.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFliteExtraManager.hpp"

namespace MNN {
namespace Express {
static void copyInfo(SpaceBatchT* dst, std::vector<VARP> inputs) {
    MNN_ASSERT(inputs.size() == 3);
    {
        auto blockShape = inputs[1];
        auto info = blockShape->getInfo();
        auto ptr = blockShape->readMap<int>();
        dst->blockShape.reset(new BlobT);
        auto block        = dst->blockShape.get();
        block->dataFormat = MNN_DATA_FORMAT_NHWC;
        block->dataType = DataType_DT_INT32;
        if (info != nullptr) {
            block->dims = info->dim;
            if (ptr != nullptr) {
                block->int32s.resize(info->size);
                ::memcpy(block->int32s.data(), ptr, info->size * sizeof(int32_t));
            }
        }
    }
    {
        auto padding = inputs[2];
        auto info = padding->getInfo();
        auto ptr = padding->readMap<int>();
        dst->padding.reset(new BlobT);
        auto block        = dst->padding.get();
        block->dataFormat = MNN_DATA_FORMAT_NHWC;
        block->dataType = DataType_DT_INT32;
        if (info != nullptr) {
            block->dims = info->dim;
            if (ptr != nullptr) {
                block->int32s.resize(info->size);
                ::memcpy(block->int32s.data(), ptr, info->size * sizeof(int32_t));
            }
        }
    }
}

class BatchToSpaceTransform : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        std::unique_ptr<OpT> bsND(new OpT);
        bsND->name       = expr->name();
        bsND->type       = OpType_BatchToSpaceND;
        bsND->main.type  = OpParameter_SpaceBatch;
        bsND->main.value = new SpaceBatchT;
        copyInfo(bsND->main.AsSpaceBatch(), inputs);
        auto newExpr = Expr::create(bsND.get(), inputs, expr->outputSize());
        return newExpr;
    }
};
class SpaceToBatchTransform : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        std::unique_ptr<OpT> bsND(new OpT);
        bsND->name       = expr->name();
        bsND->type       = OpType_SpaceToBatchND;
        bsND->main.type  = OpParameter_SpaceBatch;
        bsND->main.value = new SpaceBatchT;
        copyInfo(bsND->main.AsSpaceBatch(), inputs);
        auto newExpr = Expr::create(bsND.get(), inputs, expr->outputSize());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("BatchToSpace",
                                      std::shared_ptr<TFliteExtraManager::Transform>(new BatchToSpaceTransform));
    TFliteExtraManager::get()->insert("SpaceToBatch",
                                      std::shared_ptr<TFliteExtraManager::Transform>(new SpaceToBatchTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
