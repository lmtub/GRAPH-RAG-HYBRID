static void *aio_thread(void *unused)

{

    pid_t pid;

    sigset_t set;



    pid = getpid();



    /* block all signals */

    if (sigfillset(&set)) die("sigfillset");

    if (sigprocmask(SIG_BLOCK, &set, NULL)) die("sigprocmask");



    while (1) {

        struct qemu_paiocb *aiocb;

        size_t ret = 0;

        qemu_timeval tv;

        struct timespec ts;



        qemu_gettimeofday(&tv);

        ts.tv_sec = tv.tv_sec + 10;

        ts.tv_nsec = 0;



        mutex_lock(&lock);



        while (TAILQ_EMPTY(&request_list) &&

               !(ret == ETIMEDOUT)) {

            ret = cond_timedwait(&cond, &lock, &ts);

        }



        if (TAILQ_EMPTY(&request_list))

            break;



        aiocb = TAILQ_FIRST(&request_list);

        TAILQ_REMOVE(&request_list, aiocb, node);

        aiocb->active = 1;

        idle_threads--;

        mutex_unlock(&lock);



        switch (aiocb->aio_type) {

        case QEMU_PAIO_READ:

        case QEMU_PAIO_WRITE:

		ret = handle_aiocb_rw(aiocb);

		break;

        case QEMU_PAIO_IOCTL:

		ret = handle_aiocb_ioctl(aiocb);

		break;

	default:

		fprintf(stderr, "invalid aio request (0x%x)\n", aiocb->aio_type);

		ret = -EINVAL;

		break;

	}



        mutex_lock(&lock);

        aiocb->ret = ret;

        idle_threads++;

        mutex_unlock(&lock);



        if (kill(pid, aiocb->ev_signo)) die("kill failed");

    }



    idle_threads--;

    cur_threads--;

    mutex_unlock(&lock);



    return NULL;

}
