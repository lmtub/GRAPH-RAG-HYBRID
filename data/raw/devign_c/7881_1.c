static int qemu_rdma_get_fd(void *opaque)

{

    QEMUFileRDMA *rfile = opaque;

    RDMAContext *rdma = rfile->rdma;



    return rdma->comp_channel->fd;

}
