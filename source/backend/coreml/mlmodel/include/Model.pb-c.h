/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: Model.proto */

#ifndef PROTOBUF_C_Model_2eproto__INCLUDED
#define PROTOBUF_C_Model_2eproto__INCLUDED

#include <protobuf-c/protobuf-c.h>

PROTOBUF_C__BEGIN_DECLS

#if PROTOBUF_C_VERSION_NUMBER < 1003000
# error This file was generated by a newer version of protoc-c which is incompatible with your libprotobuf-c headers. Please update your headers.
#elif 1003003 < PROTOBUF_C_MIN_COMPILER_VERSION
# error This file was generated by an older version of protoc-c which is incompatible with your libprotobuf-c headers. Please regenerate this file with a newer version of protoc-c.
#endif

#include "NeuralNetwork.pb-c.h"

typedef struct _CoreML__Specification__FeatureDescription CoreML__Specification__FeatureDescription;
typedef struct _CoreML__Specification__Metadata CoreML__Specification__Metadata;
typedef struct _CoreML__Specification__Metadata__UserDefinedEntry CoreML__Specification__Metadata__UserDefinedEntry;
typedef struct _CoreML__Specification__ModelDescription CoreML__Specification__ModelDescription;
typedef struct _CoreML__Specification__SerializedModel CoreML__Specification__SerializedModel;
typedef struct _CoreML__Specification__Model CoreML__Specification__Model;


/* --- enums --- */


/* --- messages --- */

/*
 **
 * A feature description,
 * consisting of a name, short description, and type.
 */
struct  _CoreML__Specification__FeatureDescription
{
  ProtobufCMessage base;
  char *name;
  char *shortdescription;
  CoreML__Specification__FeatureType *type;
};
#define CORE_ML__SPECIFICATION__FEATURE_DESCRIPTION__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__feature_description__descriptor) \
    , (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string, NULL }


struct  _CoreML__Specification__Metadata__UserDefinedEntry
{
  ProtobufCMessage base;
  char *key;
  char *value;
};
#define CORE_ML__SPECIFICATION__METADATA__USER_DEFINED_ENTRY__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__metadata__user_defined_entry__descriptor) \
    , (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string }


/*
 **
 * Model metadata,
 * consisting of a short description, a version string,
 * an author, a license, and any other user defined
 * key/value meta data.
 */
struct  _CoreML__Specification__Metadata
{
  ProtobufCMessage base;
  char *shortdescription;
  char *versionstring;
  char *author;
  char *license;
  size_t n_userdefined;
  CoreML__Specification__Metadata__UserDefinedEntry **userdefined;
};
#define CORE_ML__SPECIFICATION__METADATA__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__metadata__descriptor) \
    , (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string, 0,NULL }


/*
 **
 * A description of a model,
 * consisting of descriptions of its input and output features.
 * Both regressor and classifier models require the name of the
 * primary predicted output feature (``predictedFeatureName``).
 * Classifier models can specify the output feature containing
 * probabilities for the predicted classes
 * (``predictedProbabilitiesName``).
 */
struct  _CoreML__Specification__ModelDescription
{
  ProtobufCMessage base;
  size_t n_input;
  CoreML__Specification__FeatureDescription **input;
  size_t n_output;
  CoreML__Specification__FeatureDescription **output;
  /*
   * [Required for regressor and classifier models]: the name
   * to give to an output feature containing the prediction.
   */
  char *predictedfeaturename;
  /*
   * [Optional for classifier models]: the name to give to an
   * output feature containing a dictionary mapping class
   * labels to their predicted probabilities. If not specified,
   * the dictionary will not be returned by the model.
   */
  char *predictedprobabilitiesname;
  size_t n_traininginput;
  CoreML__Specification__FeatureDescription **traininginput;
  CoreML__Specification__Metadata *metadata;
};
#define CORE_ML__SPECIFICATION__MODEL_DESCRIPTION__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__model_description__descriptor) \
    , 0,NULL, 0,NULL, (char *)protobuf_c_empty_string, (char *)protobuf_c_empty_string, 0,NULL, NULL }


struct  _CoreML__Specification__SerializedModel
{
  ProtobufCMessage base;
  /*
   * Identifier whose content describes the model type of the serialized protocol buffer message.
   */
  char *identifier;
  /*
   * Must be a valid serialized protocol buffer of the above specified type.
   */
  ProtobufCBinaryData model;
};
#define CORE_ML__SPECIFICATION__SERIALIZED_MODEL__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__serialized_model__descriptor) \
    , (char *)protobuf_c_empty_string, {0,NULL} }


typedef enum {
  CORE_ML__SPECIFICATION__MODEL__TYPE__NOT_SET = 0,
  CORE_ML__SPECIFICATION__MODEL__TYPE_NEURAL_NETWORK = 500
    PROTOBUF_C__FORCE_ENUM_TO_BE_INT_SIZE(CORE_ML__SPECIFICATION__MODEL__TYPE)
} CoreML__Specification__Model__TypeCase;

/*
 **
 * A Core ML model,
 * consisting of a specification version,
 * a model description, and a model type.
 * Core ML model compatibility is indicated by
 * a monotonically increasing specification version number,
 * which is incremented anytime a backward-incompatible change is made
 * (this is functionally equivalent to the MAJOR version number
 * described by `Semantic Versioning 2.0.0 <http://semver.org/>`_).
 * Specification Versions : OS Availability (Core ML Version)
 * 1 : iOS 11, macOS 10.13, tvOS 11, watchOS 4 (Core ML 1)
 * - Feedforward & Recurrent Neural Networks
 * - General Linear Models
 * - Tree Ensembles
 * - Support Vector Machines
 * - Pipelines
 * - Feature Engineering
 * 2 : iOS 11.2, macOS 10.13.2, tvOS 11.2, watchOS 4.2 (Core ML 1.2)
 * - Custom Layers for Neural Networks
 * - Float 16 support for Neural Network layers
 * 3 : iOS 12, macOS 10.14, tvOS 12, watchOS 5 (Core ML 2)
 * - Flexible shapes and image sizes
 * - Categorical sequences
 * - Core ML Vision Feature Print, Text Classifier, Word Tagger
 * - Non Max Suppression
 * - Crop and Resize Bilinear NN layers
 * - Custom Models
 * 4 : iOS 13, macOS 10.15, tvOS 13, watchOS 6 (Core ML 3)
 * - Updatable models
 * - Exact shape / general rank mapping for neural networks
 * - Large expansion of supported neural network layers
 *   - Generalized operations
 *   - Control flow
 *   - Dynamic layers
 *   - See NeuralNetwork.proto
 * - Nearest Neighbor Classifier
 * - Sound Analysis Prepreocessing
 * - Recommender
 * - Linked Model
 * - NLP Gazeteer
 * - NLP WordEmbedding
 * 5 : iOS 14, macOS 11, tvOS 14, watchOS 7 (Core ML 4)
 * - Model Deployment
 * - Model Encryption
 * - Unified converter API with PyTorch and Tensorflow 2 Support in coremltools 4
 * - MIL builder for neural networks and composite ops in coremltools 4
 * - New layers in neural network:
 *      - CumSum
 *      - OneHot
 *      - ClampedReLu
 *      - ArgSort
 *      - SliceBySize
 *      - Convolution3D
 *      - Pool3D
 *      - Bilinear Upsample with align corners and fractional factors
 *      - PixelShuffle
 *      - MatMul with int8 weights and int8 activations
 *      - Concat interleave
 *      - See NeuralNetwork.proto
 * - Enhanced Xcode model view with interactive previews
 * - Enhanced Xcode Playground support for Core ML models
 */
struct  _CoreML__Specification__Model
{
  ProtobufCMessage base;
  int32_t specificationversion;
  CoreML__Specification__ModelDescription *description;
  /*
   * Following model types support on-device update:
   * - NeuralNetworkClassifier
   * - NeuralNetworkRegressor
   * - NeuralNetwork
   * - KNearestNeighborsClassifier
   */
  protobuf_c_boolean isupdatable;
  CoreML__Specification__Model__TypeCase type_case;
  union {
    /*
     * generic models start at 500
     */
    CoreML__Specification__NeuralNetwork *neuralnetwork;
  };
};
#define CORE_ML__SPECIFICATION__MODEL__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&core_ml__specification__model__descriptor) \
    , 0, NULL, 0, CORE_ML__SPECIFICATION__MODEL__TYPE__NOT_SET, {0} }


/* CoreML__Specification__FeatureDescription methods */
void   core_ml__specification__feature_description__init
                     (CoreML__Specification__FeatureDescription         *message);
size_t core_ml__specification__feature_description__get_packed_size
                     (const CoreML__Specification__FeatureDescription   *message);
size_t core_ml__specification__feature_description__pack
                     (const CoreML__Specification__FeatureDescription   *message,
                      uint8_t             *out);
size_t core_ml__specification__feature_description__pack_to_buffer
                     (const CoreML__Specification__FeatureDescription   *message,
                      ProtobufCBuffer     *buffer);
CoreML__Specification__FeatureDescription *
       core_ml__specification__feature_description__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   core_ml__specification__feature_description__free_unpacked
                     (CoreML__Specification__FeatureDescription *message,
                      ProtobufCAllocator *allocator);
/* CoreML__Specification__Metadata__UserDefinedEntry methods */
void   core_ml__specification__metadata__user_defined_entry__init
                     (CoreML__Specification__Metadata__UserDefinedEntry         *message);
/* CoreML__Specification__Metadata methods */
void   core_ml__specification__metadata__init
                     (CoreML__Specification__Metadata         *message);
size_t core_ml__specification__metadata__get_packed_size
                     (const CoreML__Specification__Metadata   *message);
size_t core_ml__specification__metadata__pack
                     (const CoreML__Specification__Metadata   *message,
                      uint8_t             *out);
size_t core_ml__specification__metadata__pack_to_buffer
                     (const CoreML__Specification__Metadata   *message,
                      ProtobufCBuffer     *buffer);
CoreML__Specification__Metadata *
       core_ml__specification__metadata__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   core_ml__specification__metadata__free_unpacked
                     (CoreML__Specification__Metadata *message,
                      ProtobufCAllocator *allocator);
/* CoreML__Specification__ModelDescription methods */
void   core_ml__specification__model_description__init
                     (CoreML__Specification__ModelDescription         *message);
size_t core_ml__specification__model_description__get_packed_size
                     (const CoreML__Specification__ModelDescription   *message);
size_t core_ml__specification__model_description__pack
                     (const CoreML__Specification__ModelDescription   *message,
                      uint8_t             *out);
size_t core_ml__specification__model_description__pack_to_buffer
                     (const CoreML__Specification__ModelDescription   *message,
                      ProtobufCBuffer     *buffer);
CoreML__Specification__ModelDescription *
       core_ml__specification__model_description__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   core_ml__specification__model_description__free_unpacked
                     (CoreML__Specification__ModelDescription *message,
                      ProtobufCAllocator *allocator);
/* CoreML__Specification__SerializedModel methods */
void   core_ml__specification__serialized_model__init
                     (CoreML__Specification__SerializedModel         *message);
size_t core_ml__specification__serialized_model__get_packed_size
                     (const CoreML__Specification__SerializedModel   *message);
size_t core_ml__specification__serialized_model__pack
                     (const CoreML__Specification__SerializedModel   *message,
                      uint8_t             *out);
size_t core_ml__specification__serialized_model__pack_to_buffer
                     (const CoreML__Specification__SerializedModel   *message,
                      ProtobufCBuffer     *buffer);
CoreML__Specification__SerializedModel *
       core_ml__specification__serialized_model__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   core_ml__specification__serialized_model__free_unpacked
                     (CoreML__Specification__SerializedModel *message,
                      ProtobufCAllocator *allocator);
/* CoreML__Specification__Model methods */
void   core_ml__specification__model__init
                     (CoreML__Specification__Model         *message);
size_t core_ml__specification__model__get_packed_size
                     (const CoreML__Specification__Model   *message);
size_t core_ml__specification__model__pack
                     (const CoreML__Specification__Model   *message,
                      uint8_t             *out);
size_t core_ml__specification__model__pack_to_buffer
                     (const CoreML__Specification__Model   *message,
                      ProtobufCBuffer     *buffer);
CoreML__Specification__Model *
       core_ml__specification__model__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   core_ml__specification__model__free_unpacked
                     (CoreML__Specification__Model *message,
                      ProtobufCAllocator *allocator);
/* --- per-message closures --- */

typedef void (*CoreML__Specification__FeatureDescription_Closure)
                 (const CoreML__Specification__FeatureDescription *message,
                  void *closure_data);
typedef void (*CoreML__Specification__Metadata__UserDefinedEntry_Closure)
                 (const CoreML__Specification__Metadata__UserDefinedEntry *message,
                  void *closure_data);
typedef void (*CoreML__Specification__Metadata_Closure)
                 (const CoreML__Specification__Metadata *message,
                  void *closure_data);
typedef void (*CoreML__Specification__ModelDescription_Closure)
                 (const CoreML__Specification__ModelDescription *message,
                  void *closure_data);
typedef void (*CoreML__Specification__SerializedModel_Closure)
                 (const CoreML__Specification__SerializedModel *message,
                  void *closure_data);
typedef void (*CoreML__Specification__Model_Closure)
                 (const CoreML__Specification__Model *message,
                  void *closure_data);

/* --- services --- */


/* --- descriptors --- */

extern const ProtobufCMessageDescriptor core_ml__specification__feature_description__descriptor;
extern const ProtobufCMessageDescriptor core_ml__specification__metadata__descriptor;
extern const ProtobufCMessageDescriptor core_ml__specification__metadata__user_defined_entry__descriptor;
extern const ProtobufCMessageDescriptor core_ml__specification__model_description__descriptor;
extern const ProtobufCMessageDescriptor core_ml__specification__serialized_model__descriptor;
extern const ProtobufCMessageDescriptor core_ml__specification__model__descriptor;

PROTOBUF_C__END_DECLS


#endif  /* PROTOBUF_C_Model_2eproto__INCLUDED */
