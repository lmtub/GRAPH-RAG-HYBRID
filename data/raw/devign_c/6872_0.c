int qemu_paio_ioctl(struct qemu_paiocb *aiocb)

{

    return qemu_paio_submit(aiocb, QEMU_PAIO_IOCTL);

}
