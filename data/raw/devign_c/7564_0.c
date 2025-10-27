ssize_t qemu_paio_return(struct qemu_paiocb *aiocb)

{

    ssize_t ret;



    pthread_mutex_lock(&lock);

    ret = aiocb->ret;

    pthread_mutex_unlock(&lock);



    return ret;

}
