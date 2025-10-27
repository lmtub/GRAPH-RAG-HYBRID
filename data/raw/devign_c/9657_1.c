void qemu_co_rwlock_rdlock(CoRwlock *lock)
{
    while (lock->writer) {
        qemu_co_queue_wait(&lock->queue);
    }
    lock->reader++;
}