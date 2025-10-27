static void dma_bdrv_unmap(DMAAIOCB *dbs)

{

    int i;



    for (i = 0; i < dbs->iov.niov; ++i) {

        dma_memory_unmap(dbs->sg->as, dbs->iov.iov[i].iov_base,

                         dbs->iov.iov[i].iov_len, dbs->dir,

                         dbs->iov.iov[i].iov_len);

    }

    qemu_iovec_reset(&dbs->iov);

}
