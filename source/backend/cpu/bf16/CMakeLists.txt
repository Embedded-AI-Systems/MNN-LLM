
file(GLOB MNN_BF16_SRCS "${CMAKE_CURRENT_LIST_DIR}/*")

file(GLOB MNN_BF16_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/*")

add_library(
    MNN_BF16
    OBJECT
    ${MNN_BF16_SRCS}
    )
target_compile_options(MNN_BF16 PRIVATE -DMNN_SUPPORT_BF16)
