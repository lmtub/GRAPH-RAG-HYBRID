static size_t handle_aiocb_flush(struct qemu_paiocb *aiocb)

{

    int ret;



    ret = qemu_fdatasync(aiocb->aio_fildes);

    if (ret == -1)

        return -errno;

    return 0;

}
