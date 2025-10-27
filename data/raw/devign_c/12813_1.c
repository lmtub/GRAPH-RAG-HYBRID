void qemu_mutex_unlock(QemuMutex *mutex)

{

    assert(mutex->owner == GetCurrentThreadId());

    mutex->owner = 0;

    LeaveCriticalSection(&mutex->lock);

}
