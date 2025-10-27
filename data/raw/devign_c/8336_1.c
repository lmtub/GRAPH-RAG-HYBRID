static int qemu_rdma_get_buffer(void *opaque, uint8_t *buf,

                                int64_t pos, int size)

{

    QEMUFileRDMA *r = opaque;

    RDMAContext *rdma = r->rdma;

    RDMAControlHeader head;

    int ret = 0;



    CHECK_ERROR_STATE();



    /*

     * First, we hold on to the last SEND message we

     * were given and dish out the bytes until we run

     * out of bytes.

     */

    r->len = qemu_rdma_fill(r->rdma, buf, size, 0);

    if (r->len) {

        return r->len;

    }



    /*

     * Once we run out, we block and wait for another

     * SEND message to arrive.

     */

    ret = qemu_rdma_exchange_recv(rdma, &head, RDMA_CONTROL_QEMU_FILE);



    if (ret < 0) {

        rdma->error_state = ret;

        return ret;

    }



    /*

     * SEND was received with new bytes, now try again.

     */

    return qemu_rdma_fill(r->rdma, buf, size, 0);

}
