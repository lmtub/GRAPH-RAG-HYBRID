void qemu_sem_post(QemuSemaphore *sem)

{

    int rc;



#if defined(__APPLE__) || defined(__NetBSD__)

    pthread_mutex_lock(&sem->lock);

    if (sem->count == INT_MAX) {

        rc = EINVAL;

    } else if (sem->count++ < 0) {

        rc = pthread_cond_signal(&sem->cond);

    } else {

        rc = 0;

    }

    pthread_mutex_unlock(&sem->lock);

    if (rc != 0) {

        error_exit(rc, __func__);

    }

#else

    rc = sem_post(&sem->sem);

    if (rc < 0) {

        error_exit(errno, __func__);

    }

#endif

}
