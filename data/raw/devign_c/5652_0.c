static int aiocb_needs_copy(struct qemu_paiocb *aiocb)

{

    if (aiocb->aio_flags & QEMU_AIO_SECTOR_ALIGNED) {

        int i;



        for (i = 0; i < aiocb->aio_niov; i++)

            if ((uintptr_t) aiocb->aio_iov[i].iov_base % 512)

                return 1;

    }



    return 0;

}
