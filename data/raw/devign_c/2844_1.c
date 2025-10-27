void qemu_cond_wait(QemuCond *cond, QemuMutex *mutex)

{

    /*

     * This access is protected under the mutex.

     */

    cond->waiters++;



    /*

     * Unlock external mutex and wait for signal.

     * NOTE: we've held mutex locked long enough to increment

     * waiters count above, so there's no problem with

     * leaving mutex unlocked before we wait on semaphore.

     */

    qemu_mutex_unlock(mutex);

    WaitForSingleObject(cond->sema, INFINITE);



    /* Now waiters must rendez-vous with the signaling thread and

     * let it continue.  For cond_broadcast this has heavy contention

     * and triggers thundering herd.  So goes life.

     *

     * Decrease waiters count.  The mutex is not taken, so we have

     * to do this atomically.

     *

     * All waiters contend for the mutex at the end of this function

     * until the signaling thread relinquishes it.  To ensure

     * each waiter consumes exactly one slice of the semaphore,

     * the signaling thread stops until it is told by the last

     * waiter that it can go on.

     */

    if (InterlockedDecrement(&cond->waiters) == cond->target) {

        SetEvent(cond->continue_event);

    }



    qemu_mutex_lock(mutex);

}
