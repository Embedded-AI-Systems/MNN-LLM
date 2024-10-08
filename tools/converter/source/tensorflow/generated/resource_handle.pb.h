// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: resource_handle.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_resource_5fhandle_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_resource_5fhandle_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3019000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3019000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_resource_5fhandle_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_resource_5fhandle_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_resource_5fhandle_2eproto;
namespace tensorflow {
class ResourceHandleProto;
struct ResourceHandleProtoDefaultTypeInternal;
extern ResourceHandleProtoDefaultTypeInternal _ResourceHandleProto_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::ResourceHandleProto* Arena::CreateMaybeMessage<::tensorflow::ResourceHandleProto>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class ResourceHandleProto final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.ResourceHandleProto) */ {
 public:
  inline ResourceHandleProto() : ResourceHandleProto(nullptr) {}
  ~ResourceHandleProto() override;
  explicit constexpr ResourceHandleProto(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ResourceHandleProto(const ResourceHandleProto& from);
  ResourceHandleProto(ResourceHandleProto&& from) noexcept
    : ResourceHandleProto() {
    *this = ::std::move(from);
  }

  inline ResourceHandleProto& operator=(const ResourceHandleProto& from) {
    CopyFrom(from);
    return *this;
  }
  inline ResourceHandleProto& operator=(ResourceHandleProto&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const ResourceHandleProto& default_instance() {
    return *internal_default_instance();
  }
  static inline const ResourceHandleProto* internal_default_instance() {
    return reinterpret_cast<const ResourceHandleProto*>(
               &_ResourceHandleProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ResourceHandleProto& a, ResourceHandleProto& b) {
    a.Swap(&b);
  }
  inline void Swap(ResourceHandleProto* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ResourceHandleProto* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  ResourceHandleProto* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<ResourceHandleProto>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const ResourceHandleProto& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const ResourceHandleProto& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ResourceHandleProto* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.ResourceHandleProto";
  }
  protected:
  explicit ResourceHandleProto(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kDeviceFieldNumber = 1,
    kContainerFieldNumber = 2,
    kNameFieldNumber = 3,
    kMaybeTypeNameFieldNumber = 5,
    kHashCodeFieldNumber = 4,
  };
  // string device = 1;
  void clear_device();
  const std::string& device() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_device(ArgT0&& arg0, ArgT... args);
  std::string* mutable_device();
  PROTOBUF_NODISCARD std::string* release_device();
  void set_allocated_device(std::string* device);
  private:
  const std::string& _internal_device() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_device(const std::string& value);
  std::string* _internal_mutable_device();
  public:

  // string container = 2;
  void clear_container();
  const std::string& container() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_container(ArgT0&& arg0, ArgT... args);
  std::string* mutable_container();
  PROTOBUF_NODISCARD std::string* release_container();
  void set_allocated_container(std::string* container);
  private:
  const std::string& _internal_container() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_container(const std::string& value);
  std::string* _internal_mutable_container();
  public:

  // string name = 3;
  void clear_name();
  const std::string& name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_name();
  PROTOBUF_NODISCARD std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // string maybe_type_name = 5;
  void clear_maybe_type_name();
  const std::string& maybe_type_name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_maybe_type_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_maybe_type_name();
  PROTOBUF_NODISCARD std::string* release_maybe_type_name();
  void set_allocated_maybe_type_name(std::string* maybe_type_name);
  private:
  const std::string& _internal_maybe_type_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_maybe_type_name(const std::string& value);
  std::string* _internal_mutable_maybe_type_name();
  public:

  // uint64 hash_code = 4;
  void clear_hash_code();
  uint64_t hash_code() const;
  void set_hash_code(uint64_t value);
  private:
  uint64_t _internal_hash_code() const;
  void _internal_set_hash_code(uint64_t value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.ResourceHandleProto)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr device_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr container_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr maybe_type_name_;
  uint64_t hash_code_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_resource_5fhandle_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ResourceHandleProto

// string device = 1;
inline void ResourceHandleProto::clear_device() {
  device_.ClearToEmpty();
}
inline const std::string& ResourceHandleProto::device() const {
  // @@protoc_insertion_point(field_get:tensorflow.ResourceHandleProto.device)
  return _internal_device();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ResourceHandleProto::set_device(ArgT0&& arg0, ArgT... args) {
 
 device_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ResourceHandleProto.device)
}
inline std::string* ResourceHandleProto::mutable_device() {
  std::string* _s = _internal_mutable_device();
  // @@protoc_insertion_point(field_mutable:tensorflow.ResourceHandleProto.device)
  return _s;
}
inline const std::string& ResourceHandleProto::_internal_device() const {
  return device_.Get();
}
inline void ResourceHandleProto::_internal_set_device(const std::string& value) {
  
  device_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::_internal_mutable_device() {
  
  return device_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::release_device() {
  // @@protoc_insertion_point(field_release:tensorflow.ResourceHandleProto.device)
  return device_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ResourceHandleProto::set_allocated_device(std::string* device) {
  if (device != nullptr) {
    
  } else {
    
  }
  device_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), device,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (device_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    device_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ResourceHandleProto.device)
}

// string container = 2;
inline void ResourceHandleProto::clear_container() {
  container_.ClearToEmpty();
}
inline const std::string& ResourceHandleProto::container() const {
  // @@protoc_insertion_point(field_get:tensorflow.ResourceHandleProto.container)
  return _internal_container();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ResourceHandleProto::set_container(ArgT0&& arg0, ArgT... args) {
 
 container_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ResourceHandleProto.container)
}
inline std::string* ResourceHandleProto::mutable_container() {
  std::string* _s = _internal_mutable_container();
  // @@protoc_insertion_point(field_mutable:tensorflow.ResourceHandleProto.container)
  return _s;
}
inline const std::string& ResourceHandleProto::_internal_container() const {
  return container_.Get();
}
inline void ResourceHandleProto::_internal_set_container(const std::string& value) {
  
  container_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::_internal_mutable_container() {
  
  return container_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::release_container() {
  // @@protoc_insertion_point(field_release:tensorflow.ResourceHandleProto.container)
  return container_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ResourceHandleProto::set_allocated_container(std::string* container) {
  if (container != nullptr) {
    
  } else {
    
  }
  container_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), container,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (container_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    container_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ResourceHandleProto.container)
}

// string name = 3;
inline void ResourceHandleProto::clear_name() {
  name_.ClearToEmpty();
}
inline const std::string& ResourceHandleProto::name() const {
  // @@protoc_insertion_point(field_get:tensorflow.ResourceHandleProto.name)
  return _internal_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ResourceHandleProto::set_name(ArgT0&& arg0, ArgT... args) {
 
 name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ResourceHandleProto.name)
}
inline std::string* ResourceHandleProto::mutable_name() {
  std::string* _s = _internal_mutable_name();
  // @@protoc_insertion_point(field_mutable:tensorflow.ResourceHandleProto.name)
  return _s;
}
inline const std::string& ResourceHandleProto::_internal_name() const {
  return name_.Get();
}
inline void ResourceHandleProto::_internal_set_name(const std::string& value) {
  
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::_internal_mutable_name() {
  
  return name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::release_name() {
  // @@protoc_insertion_point(field_release:tensorflow.ResourceHandleProto.name)
  return name_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ResourceHandleProto::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (name_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ResourceHandleProto.name)
}

// uint64 hash_code = 4;
inline void ResourceHandleProto::clear_hash_code() {
  hash_code_ = uint64_t{0u};
}
inline uint64_t ResourceHandleProto::_internal_hash_code() const {
  return hash_code_;
}
inline uint64_t ResourceHandleProto::hash_code() const {
  // @@protoc_insertion_point(field_get:tensorflow.ResourceHandleProto.hash_code)
  return _internal_hash_code();
}
inline void ResourceHandleProto::_internal_set_hash_code(uint64_t value) {
  
  hash_code_ = value;
}
inline void ResourceHandleProto::set_hash_code(uint64_t value) {
  _internal_set_hash_code(value);
  // @@protoc_insertion_point(field_set:tensorflow.ResourceHandleProto.hash_code)
}

// string maybe_type_name = 5;
inline void ResourceHandleProto::clear_maybe_type_name() {
  maybe_type_name_.ClearToEmpty();
}
inline const std::string& ResourceHandleProto::maybe_type_name() const {
  // @@protoc_insertion_point(field_get:tensorflow.ResourceHandleProto.maybe_type_name)
  return _internal_maybe_type_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ResourceHandleProto::set_maybe_type_name(ArgT0&& arg0, ArgT... args) {
 
 maybe_type_name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ResourceHandleProto.maybe_type_name)
}
inline std::string* ResourceHandleProto::mutable_maybe_type_name() {
  std::string* _s = _internal_mutable_maybe_type_name();
  // @@protoc_insertion_point(field_mutable:tensorflow.ResourceHandleProto.maybe_type_name)
  return _s;
}
inline const std::string& ResourceHandleProto::_internal_maybe_type_name() const {
  return maybe_type_name_.Get();
}
inline void ResourceHandleProto::_internal_set_maybe_type_name(const std::string& value) {
  
  maybe_type_name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::_internal_mutable_maybe_type_name() {
  
  return maybe_type_name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ResourceHandleProto::release_maybe_type_name() {
  // @@protoc_insertion_point(field_release:tensorflow.ResourceHandleProto.maybe_type_name)
  return maybe_type_name_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ResourceHandleProto::set_allocated_maybe_type_name(std::string* maybe_type_name) {
  if (maybe_type_name != nullptr) {
    
  } else {
    
  }
  maybe_type_name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), maybe_type_name,
      GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (maybe_type_name_.IsDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited())) {
    maybe_type_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ResourceHandleProto.maybe_type_name)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_resource_5fhandle_2eproto
