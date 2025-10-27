static void *aio_thread(void *unused)

{

    sigset_t set;



    /* block all signals */

    sigfillset(&set);

    sigprocmask(SIG_BLOCK, &set, NULL);



    while (1) {

        struct qemu_paiocb *aiocb;

        size_t offset;

        int ret = 0;



        pthread_mutex_lock(&lock);



        while (TAILQ_EMPTY(&request_list) &&

               !(ret == ETIMEDOUT)) {

            struct timespec ts = { 0 };

            qemu_timeval tv;



            qemu_gettimeofday(&tv);

            ts.tv_sec = tv.tv_sec + 10;

            ret = pthread_cond_timedwait(&cond, &lock, &ts);

        }



        if (ret == ETIMEDOUT)

            break;



        aiocb = TAILQ_FIRST(&request_list);

        TAILQ_REMOVE(&request_list, aiocb, node);



        offset = 0;

        aiocb->active = 1;



        idle_threads--;

        pthread_mutex_unlock(&lock);



        while (offset < aiocb->aio_nbytes) {

            ssize_t len;



            if (aiocb->is_write)

                len = pwrite(aiocb->aio_fildes,

                             (const char *)aiocb->aio_buf + offset,

                             aiocb->aio_nbytes - offset,

                             aiocb->aio_offset + offset);

            else

                len = pread(aiocb->aio_fildes,

                            (char *)aiocb->aio_buf + offset,

                            aiocb->aio_nbytes - offset,

                            aiocb->aio_offset + offset);



            if (len == -1 && errno == EINTR)

                continue;

            else if (len == -1) {

                pthread_mutex_lock(&lock);

                aiocb->ret = -errno;

                pthread_mutex_unlock(&lock);

                break;

            } else if (len == 0)

                break;



            offset += len;



            pthread_mutex_lock(&lock);

            aiocb->ret = offset;

            pthread_mutex_unlock(&lock);

        }



        pthread_mutex_lock(&lock);

        idle_threads++;

        pthread_mutex_unlock(&lock);



        sigqueue(getpid(),

                 aiocb->aio_sigevent.sigev_signo,

                 aiocb->aio_sigevent.sigev_value);

    }



    idle_threads--;

    cur_threads--;

    pthread_mutex_unlock(&lock);



    return NULL;

}
