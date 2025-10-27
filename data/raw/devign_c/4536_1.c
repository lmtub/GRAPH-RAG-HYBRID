static int qemu_rdma_close(void *opaque)

{

    DPRINTF("Shutting down connection.\n");

    QEMUFileRDMA *r = opaque;

    if (r->rdma) {

        qemu_rdma_cleanup(r->rdma);

        g_free(r->rdma);

    }

    g_free(r);

    return 0;

}
