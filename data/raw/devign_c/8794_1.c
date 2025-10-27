static int default_lockmgr_cb(void **arg, enum AVLockOp op)

{

    void * volatile * mutex = arg;

    int err;



    switch (op) {

    case AV_LOCK_CREATE:

        return 0;

    case AV_LOCK_OBTAIN:

        if (!*mutex) {

            pthread_mutex_t *tmp = av_malloc(sizeof(pthread_mutex_t));

            if (!tmp)

                return AVERROR(ENOMEM);

            if ((err = pthread_mutex_init(tmp, NULL))) {

                av_free(tmp);

                return AVERROR(err);

            }

            if (avpriv_atomic_ptr_cas(mutex, NULL, tmp)) {

                pthread_mutex_destroy(tmp);

                av_free(tmp);

            }

        }



        if ((err = pthread_mutex_lock(*mutex)))

            return AVERROR(err);



        return 0;

    case AV_LOCK_RELEASE:

        if ((err = pthread_mutex_unlock(*mutex)))

            return AVERROR(err);



        return 0;

    case AV_LOCK_DESTROY:

        if (*mutex)

            pthread_mutex_destroy(*mutex);

        av_free(*mutex);

        avpriv_atomic_ptr_cas(mutex, *mutex, NULL);

        return 0;

    }

    return 1;

}
