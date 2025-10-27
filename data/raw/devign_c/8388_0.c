static int qemu_rdma_drain_cq(QEMUFile *f, RDMAContext *rdma)

{

    int ret;



    if (qemu_rdma_write_flush(f, rdma) < 0) {

        return -EIO;

    }



    while (rdma->nb_sent) {

        ret = qemu_rdma_block_for_wrid(rdma, RDMA_WRID_RDMA_WRITE);

        if (ret < 0) {

            fprintf(stderr, "rdma migration: complete polling error!\n");

            return -EIO;

        }

    }



    qemu_rdma_unregister_waiting(rdma);



    return 0;

}
