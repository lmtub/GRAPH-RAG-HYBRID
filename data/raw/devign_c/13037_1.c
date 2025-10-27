void qemu_mutex_lock(QemuMutex *mutex)

{

    EnterCriticalSection(&mutex->lock);



    /* Win32 CRITICAL_SECTIONs are recursive.  Assert that we're not

     * using them as such.

     */

    assert(mutex->owner == 0);

    mutex->owner = GetCurrentThreadId();

}
