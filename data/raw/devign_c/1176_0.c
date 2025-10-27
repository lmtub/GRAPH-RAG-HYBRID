int qemu_paio_error(struct qemu_paiocb *aiocb)

{

    ssize_t ret = qemu_paio_return(aiocb);



    if (ret < 0)

        ret = -ret;

    else

        ret = 0;



    return ret;

}
