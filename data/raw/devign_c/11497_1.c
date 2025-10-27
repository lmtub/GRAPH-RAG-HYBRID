static int qemu_rdma_post_send_control(RDMAContext *rdma, uint8_t *buf,

                                       RDMAControlHeader *head)

{

    int ret = 0;

    RDMAWorkRequestData *wr = &rdma->wr_data[RDMA_WRID_CONTROL];

    struct ibv_send_wr *bad_wr;

    struct ibv_sge sge = {

                           .addr = (uint64_t)(wr->control),

                           .length = head->len + sizeof(RDMAControlHeader),

                           .lkey = wr->control_mr->lkey,

                         };

    struct ibv_send_wr send_wr = {

                                   .wr_id = RDMA_WRID_SEND_CONTROL,

                                   .opcode = IBV_WR_SEND,

                                   .send_flags = IBV_SEND_SIGNALED,

                                   .sg_list = &sge,

                                   .num_sge = 1,

                                };



    DDDPRINTF("CONTROL: sending %s..\n", control_desc[head->type]);



    /*

     * We don't actually need to do a memcpy() in here if we used

     * the "sge" properly, but since we're only sending control messages

     * (not RAM in a performance-critical path), then its OK for now.

     *

     * The copy makes the RDMAControlHeader simpler to manipulate

     * for the time being.

     */

    assert(head->len <= RDMA_CONTROL_MAX_BUFFER - sizeof(*head));

    memcpy(wr->control, head, sizeof(RDMAControlHeader));

    control_to_network((void *) wr->control);



    if (buf) {

        memcpy(wr->control + sizeof(RDMAControlHeader), buf, head->len);

    }





    ret = ibv_post_send(rdma->qp, &send_wr, &bad_wr);



    if (ret > 0) {

        fprintf(stderr, "Failed to use post IB SEND for control!\n");

        return -ret;

    }



    ret = qemu_rdma_block_for_wrid(rdma, RDMA_WRID_SEND_CONTROL, NULL);

    if (ret < 0) {

        fprintf(stderr, "rdma migration: send polling control error!\n");

    }



    return ret;

}
