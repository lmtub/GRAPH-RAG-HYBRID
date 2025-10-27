int qemu_sem_timedwait(QemuSemaphore *sem, int ms)

{

    int rc = WaitForSingleObject(sem->sema, ms);

    if (rc == WAIT_OBJECT_0) {

        return 0;

    }

    if (rc != WAIT_TIMEOUT) {

        error_exit(GetLastError(), __func__);

    }

    return -1;

}
