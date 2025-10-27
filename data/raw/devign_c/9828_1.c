void qemu_cond_broadcast(QemuCond *cond)

{

    BOOLEAN result;

    /*

     * As in pthread_cond_signal, access to cond->waiters and

     * cond->target is locked via the external mutex.

     */

    if (cond->waiters == 0) {

        return;

    }



    cond->target = 0;

    result = ReleaseSemaphore(cond->sema, cond->waiters, NULL);

    if (!result) {

        error_exit(GetLastError(), __func__);

    }



    /*

     * At this point all waiters continue. Each one takes its

     * slice of the semaphore. Now it's our turn to wait: Since

     * the external mutex is held, no thread can leave cond_wait,

     * yet. For this reason, we can be sure that no thread gets

     * a chance to eat *more* than one slice. OTOH, it means

     * that the last waiter must send us a wake-up.

     */

    WaitForSingleObject(cond->continue_event, INFINITE);

}
