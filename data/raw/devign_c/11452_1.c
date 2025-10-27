void qemu_mutex_init(QemuMutex *mutex)

{

    int err;

    pthread_mutexattr_t mutexattr;



    pthread_mutexattr_init(&mutexattr);

    pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_ERRORCHECK);

    err = pthread_mutex_init(&mutex->lock, &mutexattr);

    pthread_mutexattr_destroy(&mutexattr);

    if (err)

        error_exit(err, __func__);

}
