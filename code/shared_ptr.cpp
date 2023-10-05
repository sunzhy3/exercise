#include <iostream>
#include <mutex>
#include <thread>
using namespace std;

template<class T>
class Shared_Ptr {
 public:
    Shared_Ptr(T* ptr = nullptr)
        :_pPtr(ptr)
        , _pRefCount(new int(1))
        , _pMutex(new mutex) {}

    ~Shared_Ptr() {
        Release();
    }
 
    Shared_Ptr(const Shared_Ptr<T>& sp)
        :_pPtr(sp._pPtr)
        , _pRefCount(sp._pRefCount)
        , _pMutex(sp._pMutex) {
        AddRefCount();
    }

    Shared_Ptr<T>& operator=(const Shared_Ptr<T>& sp) {
        if (_pPtr != sp._pPtr) {
            // 释放管理的旧资源
            Release();
            // 共享管理新对象的资源，并增加引用计数
            _pPtr = sp._pPtr;
            _pRefCount = sp._pRefCount;
            _pMutex = sp._pMutex;
            AddRefCount();
        }
        return *this;
    }

    T& operator*() {
        return *_pPtr;
    }

    T* operator->() {
        return _pPtr;
    }

    int UseCount() { 
        return *_pRefCount; 
    }

    T* Get() { 
        return _pPtr; 
    }

    void AddRefCount() {
        _pMutex->lock();
        ++(*_pRefCount);
        _pMutex->unlock();
    }

private:
    void Release() {
        bool deleteflag = false;
        _pMutex->lock();
        if (--(*_pRefCount) == 0) {
            delete _pRefCount;
            delete _pPtr;
            deleteflag = true;
        }
        _pMutex->unlock();
        if (deleteflag == true)
            delete _pMutex;
    }

    int *_pRefCount;
    T* _pPtr;
    mutex* _pMutex;
};