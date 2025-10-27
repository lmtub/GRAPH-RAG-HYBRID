int qemu_paio_cancel(int fd, struct qemu_paiocb *aiocb)

{

    int ret;



    pthread_mutex_lock(&lock);

    if (!aiocb->active) {

        TAILQ_REMOVE(&request_list, aiocb, node);

        aiocb->ret = -ECANCELED;

        ret = QEMU_PAIO_CANCELED;

    } else if (aiocb->ret == -EINPROGRESS)

        ret = QEMU_PAIO_NOTCANCELED;

    else

        ret = QEMU_PAIO_ALLDONE;

    pthread_mutex_unlock(&lock);



    return ret;

}
