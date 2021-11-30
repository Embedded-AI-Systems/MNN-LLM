// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_METALCACHE_METALCACHE_H_
#define FLATBUFFERS_GENERATED_METALCACHE_METALCACHE_H_

#include "flatbuffers/flatbuffers.h"

namespace MetalCache {

struct Autotuning;
struct AutotuningT;

struct Cache;
struct CacheT;

struct AutotuningT : public flatbuffers::NativeTable {
  typedef Autotuning TableType;
  std::string key;
  std::vector<uint32_t> threadSize;
  std::vector<uint32_t> groupNum;
  std::vector<uint32_t> groupSize;
  uint32_t timeCost;
  AutotuningT()
      : timeCost(0) {
  }
};

struct Autotuning FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef AutotuningT NativeTableType;
  const flatbuffers::String *key() const {
    return GetPointer<const flatbuffers::String *>(4);
  }
  const flatbuffers::Vector<uint32_t> *threadSize() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(6);
  }
  const flatbuffers::Vector<uint32_t> *groupNum() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(8);
  }
  const flatbuffers::Vector<uint32_t> *groupSize() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(10);
  }
  uint32_t timeCost() const {
    return GetField<uint32_t>(12, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, 4) &&
           verifier.VerifyString(key()) &&
           VerifyOffset(verifier, 6) &&
           verifier.VerifyVector(threadSize()) &&
           VerifyOffset(verifier, 8) &&
           verifier.VerifyVector(groupNum()) &&
           VerifyOffset(verifier, 10) &&
           verifier.VerifyVector(groupSize()) &&
           VerifyField<uint32_t>(verifier, 12) &&
           verifier.EndTable();
  }
  AutotuningT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(AutotuningT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Autotuning> Pack(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct AutotuningBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_key(flatbuffers::Offset<flatbuffers::String> key) {
    fbb_.AddOffset(4, key);
  }
  void add_threadSize(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> threadSize) {
    fbb_.AddOffset(6, threadSize);
  }
  void add_groupNum(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> groupNum) {
    fbb_.AddOffset(8, groupNum);
  }
  void add_groupSize(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> groupSize) {
    fbb_.AddOffset(10, groupSize);
  }
  void add_timeCost(uint32_t timeCost) {
    fbb_.AddElement<uint32_t>(12, timeCost, 0);
  }
  explicit AutotuningBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  AutotuningBuilder &operator=(const AutotuningBuilder &);
  flatbuffers::Offset<Autotuning> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Autotuning>(end);
    return o;
  }
};

inline flatbuffers::Offset<Autotuning> CreateAutotuning(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> key = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> threadSize = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> groupNum = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> groupSize = 0,
    uint32_t timeCost = 0) {
  AutotuningBuilder builder_(_fbb);
  builder_.add_timeCost(timeCost);
  builder_.add_groupSize(groupSize);
  builder_.add_groupNum(groupNum);
  builder_.add_threadSize(threadSize);
  builder_.add_key(key);
  return builder_.Finish();
}

flatbuffers::Offset<Autotuning> CreateAutotuning(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct CacheT : public flatbuffers::NativeTable {
  typedef Cache TableType;
  std::vector<std::unique_ptr<AutotuningT>> tunings;
  CacheT() {
  }
};

struct Cache FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef CacheT NativeTableType;
  const flatbuffers::Vector<flatbuffers::Offset<Autotuning>> *tunings() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Autotuning>> *>(4);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, 4) &&
           verifier.VerifyVector(tunings()) &&
           verifier.VerifyVectorOfTables(tunings()) &&
           verifier.EndTable();
  }
  CacheT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(CacheT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Cache> Pack(flatbuffers::FlatBufferBuilder &_fbb, const CacheT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct CacheBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_tunings(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Autotuning>>> tunings) {
    fbb_.AddOffset(4, tunings);
  }
  explicit CacheBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  CacheBuilder &operator=(const CacheBuilder &);
  flatbuffers::Offset<Cache> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Cache>(end);
    return o;
  }
};

inline flatbuffers::Offset<Cache> CreateCache(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Autotuning>>> tunings = 0) {
  CacheBuilder builder_(_fbb);
  builder_.add_tunings(tunings);
  return builder_.Finish();
}

flatbuffers::Offset<Cache> CreateCache(flatbuffers::FlatBufferBuilder &_fbb, const CacheT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline AutotuningT *Autotuning::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new AutotuningT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void Autotuning::UnPackTo(AutotuningT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = key(); if (_e) _o->key = _e->str(); };
  { auto _e = threadSize(); if (_e) { _o->threadSize.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->threadSize[_i] = _e->Get(_i); } } };
  { auto _e = groupNum(); if (_e) { _o->groupNum.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->groupNum[_i] = _e->Get(_i); } } };
  { auto _e = groupSize(); if (_e) { _o->groupSize.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->groupSize[_i] = _e->Get(_i); } } };
  { auto _e = timeCost(); _o->timeCost = _e; };
}

inline flatbuffers::Offset<Autotuning> Autotuning::Pack(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateAutotuning(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Autotuning> CreateAutotuning(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const AutotuningT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _key = _o->key.empty() ? 0 : _fbb.CreateString(_o->key);
  auto _threadSize = _o->threadSize.size() ? _fbb.CreateVector(_o->threadSize) : 0;
  auto _groupNum = _o->groupNum.size() ? _fbb.CreateVector(_o->groupNum) : 0;
  auto _groupSize = _o->groupSize.size() ? _fbb.CreateVector(_o->groupSize) : 0;
  auto _timeCost = _o->timeCost;
  return MetalCache::CreateAutotuning(
      _fbb,
      _key,
      _threadSize,
      _groupNum,
      _groupSize,
      _timeCost);
}

inline CacheT *Cache::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new CacheT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void Cache::UnPackTo(CacheT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = tunings(); if (_e) { _o->tunings.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->tunings[_i] = std::unique_ptr<AutotuningT>(_e->Get(_i)->UnPack(_resolver)); } } };
}

inline flatbuffers::Offset<Cache> Cache::Pack(flatbuffers::FlatBufferBuilder &_fbb, const CacheT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateCache(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Cache> CreateCache(flatbuffers::FlatBufferBuilder &_fbb, const CacheT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const CacheT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _tunings = _o->tunings.size() ? _fbb.CreateVector<flatbuffers::Offset<Autotuning>> (_o->tunings.size(), [](size_t i, _VectorArgs *__va) { return CreateAutotuning(*__va->__fbb, __va->__o->tunings[i].get(), __va->__rehasher); }, &_va ) : 0;
  return MetalCache::CreateCache(
      _fbb,
      _tunings);
}

inline const MetalCache::Cache *GetCache(const void *buf) {
  return flatbuffers::GetRoot<MetalCache::Cache>(buf);
}

inline const MetalCache::Cache *GetSizePrefixedCache(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<MetalCache::Cache>(buf);
}

inline bool VerifyCacheBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<MetalCache::Cache>(nullptr);
}

inline bool VerifySizePrefixedCacheBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<MetalCache::Cache>(nullptr);
}

inline void FinishCacheBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<MetalCache::Cache> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedCacheBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<MetalCache::Cache> root) {
  fbb.FinishSizePrefixed(root);
}

inline std::unique_ptr<CacheT> UnPackCache(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<CacheT>(GetCache(buf)->UnPack(res));
}

}  // namespace MetalCache

#endif  // FLATBUFFERS_GENERATED_METALCACHE_METALCACHE_H_