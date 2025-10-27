static int qemu_paio_submit(struct qemu_paiocb *aiocb, int is_write)

{

    aiocb->is_write = is_write;

    aiocb->ret = -EINPROGRESS;

    aiocb->active = 0;

    mutex_lock(&lock);

    if (idle_threads == 0 && cur_threads < max_threads)

        spawn_thread();

    TAILQ_INSERT_TAIL(&request_list, aiocb, node);

    mutex_unlock(&lock);

    cond_broadcast(&cond);



    return 0;

}
