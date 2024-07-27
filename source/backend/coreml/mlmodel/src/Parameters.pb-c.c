/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: Parameters.proto */

/* Do not generate deprecated warnings for self */
#ifndef PROTOBUF_C__NO_DEPRECATED
#define PROTOBUF_C__NO_DEPRECATED
#endif

#include "Parameters.pb-c.h"
void   core_ml__specification__int64_parameter__init
                     (CoreML__Specification__Int64Parameter         *message)
{
  static const CoreML__Specification__Int64Parameter init_value = CORE_ML__SPECIFICATION__INT64_PARAMETER__INIT;
  *message = init_value;
}
size_t core_ml__specification__int64_parameter__get_packed_size
                     (const CoreML__Specification__Int64Parameter *message)
{
  assert(message->base.descriptor == &core_ml__specification__int64_parameter__descriptor);
  return protobuf_c_message_get_packed_size ((const ProtobufCMessage*)(message));
}
size_t core_ml__specification__int64_parameter__pack
                     (const CoreML__Specification__Int64Parameter *message,
                      uint8_t       *out)
{
  assert(message->base.descriptor == &core_ml__specification__int64_parameter__descriptor);
  return protobuf_c_message_pack ((const ProtobufCMessage*)message, out);
}
size_t core_ml__specification__int64_parameter__pack_to_buffer
                     (const CoreML__Specification__Int64Parameter *message,
                      ProtobufCBuffer *buffer)
{
  assert(message->base.descriptor == &core_ml__specification__int64_parameter__descriptor);
  return protobuf_c_message_pack_to_buffer ((const ProtobufCMessage*)message, buffer);
}
CoreML__Specification__Int64Parameter *
       core_ml__specification__int64_parameter__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data)
{
  return (CoreML__Specification__Int64Parameter *)
     protobuf_c_message_unpack (&core_ml__specification__int64_parameter__descriptor,
                                allocator, len, data);
}
void   core_ml__specification__int64_parameter__free_unpacked
                     (CoreML__Specification__Int64Parameter *message,
                      ProtobufCAllocator *allocator)
{
  if(!message)
    return;
  assert(message->base.descriptor == &core_ml__specification__int64_parameter__descriptor);
  protobuf_c_message_free_unpacked ((ProtobufCMessage*)message, allocator);
}
void   core_ml__specification__double_parameter__init
                     (CoreML__Specification__DoubleParameter         *message)
{
  static const CoreML__Specification__DoubleParameter init_value = CORE_ML__SPECIFICATION__DOUBLE_PARAMETER__INIT;
  *message = init_value;
}
size_t core_ml__specification__double_parameter__get_packed_size
                     (const CoreML__Specification__DoubleParameter *message)
{
  assert(message->base.descriptor == &core_ml__specification__double_parameter__descriptor);
  return protobuf_c_message_get_packed_size ((const ProtobufCMessage*)(message));
}
size_t core_ml__specification__double_parameter__pack
                     (const CoreML__Specification__DoubleParameter *message,
                      uint8_t       *out)
{
  assert(message->base.descriptor == &core_ml__specification__double_parameter__descriptor);
  return protobuf_c_message_pack ((const ProtobufCMessage*)message, out);
}
size_t core_ml__specification__double_parameter__pack_to_buffer
                     (const CoreML__Specification__DoubleParameter *message,
                      ProtobufCBuffer *buffer)
{
  assert(message->base.descriptor == &core_ml__specification__double_parameter__descriptor);
  return protobuf_c_message_pack_to_buffer ((const ProtobufCMessage*)message, buffer);
}
CoreML__Specification__DoubleParameter *
       core_ml__specification__double_parameter__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data)
{
  return (CoreML__Specification__DoubleParameter *)
     protobuf_c_message_unpack (&core_ml__specification__double_parameter__descriptor,
                                allocator, len, data);
}
void   core_ml__specification__double_parameter__free_unpacked
                     (CoreML__Specification__DoubleParameter *message,
                      ProtobufCAllocator *allocator)
{
  if(!message)
    return;
  assert(message->base.descriptor == &core_ml__specification__double_parameter__descriptor);
  protobuf_c_message_free_unpacked ((ProtobufCMessage*)message, allocator);
}
void   core_ml__specification__string_parameter__init
                     (CoreML__Specification__StringParameter         *message)
{
  static const CoreML__Specification__StringParameter init_value = CORE_ML__SPECIFICATION__STRING_PARAMETER__INIT;
  *message = init_value;
}
size_t core_ml__specification__string_parameter__get_packed_size
                     (const CoreML__Specification__StringParameter *message)
{
  assert(message->base.descriptor == &core_ml__specification__string_parameter__descriptor);
  return protobuf_c_message_get_packed_size ((const ProtobufCMessage*)(message));
}
size_t core_ml__specification__string_parameter__pack
                     (const CoreML__Specification__StringParameter *message,
                      uint8_t       *out)
{
  assert(message->base.descriptor == &core_ml__specification__string_parameter__descriptor);
  return protobuf_c_message_pack ((const ProtobufCMessage*)message, out);
}
size_t core_ml__specification__string_parameter__pack_to_buffer
                     (const CoreML__Specification__StringParameter *message,
                      ProtobufCBuffer *buffer)
{
  assert(message->base.descriptor == &core_ml__specification__string_parameter__descriptor);
  return protobuf_c_message_pack_to_buffer ((const ProtobufCMessage*)message, buffer);
}
CoreML__Specification__StringParameter *
       core_ml__specification__string_parameter__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data)
{
  return (CoreML__Specification__StringParameter *)
     protobuf_c_message_unpack (&core_ml__specification__string_parameter__descriptor,
                                allocator, len, data);
}
void   core_ml__specification__string_parameter__free_unpacked
                     (CoreML__Specification__StringParameter *message,
                      ProtobufCAllocator *allocator)
{
  if(!message)
    return;
  assert(message->base.descriptor == &core_ml__specification__string_parameter__descriptor);
  protobuf_c_message_free_unpacked ((ProtobufCMessage*)message, allocator);
}
void   core_ml__specification__bool_parameter__init
                     (CoreML__Specification__BoolParameter         *message)
{
  static const CoreML__Specification__BoolParameter init_value = CORE_ML__SPECIFICATION__BOOL_PARAMETER__INIT;
  *message = init_value;
}
size_t core_ml__specification__bool_parameter__get_packed_size
                     (const CoreML__Specification__BoolParameter *message)
{
  assert(message->base.descriptor == &core_ml__specification__bool_parameter__descriptor);
  return protobuf_c_message_get_packed_size ((const ProtobufCMessage*)(message));
}
size_t core_ml__specification__bool_parameter__pack
                     (const CoreML__Specification__BoolParameter *message,
                      uint8_t       *out)
{
  assert(message->base.descriptor == &core_ml__specification__bool_parameter__descriptor);
  return protobuf_c_message_pack ((const ProtobufCMessage*)message, out);
}
size_t core_ml__specification__bool_parameter__pack_to_buffer
                     (const CoreML__Specification__BoolParameter *message,
                      ProtobufCBuffer *buffer)
{
  assert(message->base.descriptor == &core_ml__specification__bool_parameter__descriptor);
  return protobuf_c_message_pack_to_buffer ((const ProtobufCMessage*)message, buffer);
}
CoreML__Specification__BoolParameter *
       core_ml__specification__bool_parameter__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data)
{
  return (CoreML__Specification__BoolParameter *)
     protobuf_c_message_unpack (&core_ml__specification__bool_parameter__descriptor,
                                allocator, len, data);
}
void   core_ml__specification__bool_parameter__free_unpacked
                     (CoreML__Specification__BoolParameter *message,
                      ProtobufCAllocator *allocator)
{
  if(!message)
    return;
  assert(message->base.descriptor == &core_ml__specification__bool_parameter__descriptor);
  protobuf_c_message_free_unpacked ((ProtobufCMessage*)message, allocator);
}
static const ProtobufCFieldDescriptor core_ml__specification__int64_parameter__field_descriptors[3] =
{
  {
    "defaultValue",
    1,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_INT64,
    0,   /* quantifier_offset */
    offsetof(CoreML__Specification__Int64Parameter, defaultvalue),
    NULL,
    NULL,
    0,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
  {
    "range",
    10,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_MESSAGE,
    offsetof(CoreML__Specification__Int64Parameter, allowed_values_case),
    offsetof(CoreML__Specification__Int64Parameter, range),
    &core_ml__specification__int64_range__descriptor,
    NULL,
    0 | PROTOBUF_C_FIELD_FLAG_ONEOF,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
  {
    "set",
    11,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_MESSAGE,
    offsetof(CoreML__Specification__Int64Parameter, allowed_values_case),
    offsetof(CoreML__Specification__Int64Parameter, set),
    &core_ml__specification__int64_set__descriptor,
    NULL,
    0 | PROTOBUF_C_FIELD_FLAG_ONEOF,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
};
static const unsigned core_ml__specification__int64_parameter__field_indices_by_name[] = {
  0,   /* field[0] = defaultValue */
  1,   /* field[1] = range */
  2,   /* field[2] = set */
};
static const ProtobufCIntRange core_ml__specification__int64_parameter__number_ranges[2 + 1] =
{
  { 1, 0 },
  { 10, 1 },
  { 0, 3 }
};
const ProtobufCMessageDescriptor core_ml__specification__int64_parameter__descriptor =
{
  PROTOBUF_C__MESSAGE_DESCRIPTOR_MAGIC,
  "CoreML.Specification.Int64Parameter",
  "Int64Parameter",
  "CoreML__Specification__Int64Parameter",
  "CoreML.Specification",
  sizeof(CoreML__Specification__Int64Parameter),
  3,
  core_ml__specification__int64_parameter__field_descriptors,
  core_ml__specification__int64_parameter__field_indices_by_name,
  2,  core_ml__specification__int64_parameter__number_ranges,
  (ProtobufCMessageInit) core_ml__specification__int64_parameter__init,
  NULL,NULL,NULL    /* reserved[123] */
};
static const ProtobufCFieldDescriptor core_ml__specification__double_parameter__field_descriptors[2] =
{
  {
    "defaultValue",
    1,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_DOUBLE,
    0,   /* quantifier_offset */
    offsetof(CoreML__Specification__DoubleParameter, defaultvalue),
    NULL,
    NULL,
    0,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
  {
    "range",
    10,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_MESSAGE,
    offsetof(CoreML__Specification__DoubleParameter, allowed_values_case),
    offsetof(CoreML__Specification__DoubleParameter, range),
    &core_ml__specification__double_range__descriptor,
    NULL,
    0 | PROTOBUF_C_FIELD_FLAG_ONEOF,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
};
static const unsigned core_ml__specification__double_parameter__field_indices_by_name[] = {
  0,   /* field[0] = defaultValue */
  1,   /* field[1] = range */
};
static const ProtobufCIntRange core_ml__specification__double_parameter__number_ranges[2 + 1] =
{
  { 1, 0 },
  { 10, 1 },
  { 0, 2 }
};
const ProtobufCMessageDescriptor core_ml__specification__double_parameter__descriptor =
{
  PROTOBUF_C__MESSAGE_DESCRIPTOR_MAGIC,
  "CoreML.Specification.DoubleParameter",
  "DoubleParameter",
  "CoreML__Specification__DoubleParameter",
  "CoreML.Specification",
  sizeof(CoreML__Specification__DoubleParameter),
  2,
  core_ml__specification__double_parameter__field_descriptors,
  core_ml__specification__double_parameter__field_indices_by_name,
  2,  core_ml__specification__double_parameter__number_ranges,
  (ProtobufCMessageInit) core_ml__specification__double_parameter__init,
  NULL,NULL,NULL    /* reserved[123] */
};
static const ProtobufCFieldDescriptor core_ml__specification__string_parameter__field_descriptors[1] =
{
  {
    "defaultValue",
    1,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_STRING,
    0,   /* quantifier_offset */
    offsetof(CoreML__Specification__StringParameter, defaultvalue),
    NULL,
    &protobuf_c_empty_string,
    0,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
};
static const unsigned core_ml__specification__string_parameter__field_indices_by_name[] = {
  0,   /* field[0] = defaultValue */
};
static const ProtobufCIntRange core_ml__specification__string_parameter__number_ranges[1 + 1] =
{
  { 1, 0 },
  { 0, 1 }
};
const ProtobufCMessageDescriptor core_ml__specification__string_parameter__descriptor =
{
  PROTOBUF_C__MESSAGE_DESCRIPTOR_MAGIC,
  "CoreML.Specification.StringParameter",
  "StringParameter",
  "CoreML__Specification__StringParameter",
  "CoreML.Specification",
  sizeof(CoreML__Specification__StringParameter),
  1,
  core_ml__specification__string_parameter__field_descriptors,
  core_ml__specification__string_parameter__field_indices_by_name,
  1,  core_ml__specification__string_parameter__number_ranges,
  (ProtobufCMessageInit) core_ml__specification__string_parameter__init,
  NULL,NULL,NULL    /* reserved[123] */
};
static const ProtobufCFieldDescriptor core_ml__specification__bool_parameter__field_descriptors[1] =
{
  {
    "defaultValue",
    1,
    PROTOBUF_C_LABEL_NONE,
    PROTOBUF_C_TYPE_BOOL,
    0,   /* quantifier_offset */
    offsetof(CoreML__Specification__BoolParameter, defaultvalue),
    NULL,
    NULL,
    0,             /* flags */
    0,NULL,NULL    /* reserved1,reserved2, etc */
  },
};
static const unsigned core_ml__specification__bool_parameter__field_indices_by_name[] = {
  0,   /* field[0] = defaultValue */
};
static const ProtobufCIntRange core_ml__specification__bool_parameter__number_ranges[1 + 1] =
{
  { 1, 0 },
  { 0, 1 }
};
const ProtobufCMessageDescriptor core_ml__specification__bool_parameter__descriptor =
{
  PROTOBUF_C__MESSAGE_DESCRIPTOR_MAGIC,
  "CoreML.Specification.BoolParameter",
  "BoolParameter",
  "CoreML__Specification__BoolParameter",
  "CoreML.Specification",
  sizeof(CoreML__Specification__BoolParameter),
  1,
  core_ml__specification__bool_parameter__field_descriptors,
  core_ml__specification__bool_parameter__field_indices_by_name,
  1,  core_ml__specification__bool_parameter__number_ranges,
  (ProtobufCMessageInit) core_ml__specification__bool_parameter__init,
  NULL,NULL,NULL    /* reserved[123] */
};
