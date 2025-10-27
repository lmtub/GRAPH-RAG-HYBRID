static int qemu_rdma_exchange_recv(RDMAContext *rdma, RDMAControlHeader *head,

                                int expecting)

{

    RDMAControlHeader ready = {

                                .len = 0,

                                .type = RDMA_CONTROL_READY,

                                .repeat = 1,

                              };

    int ret;



    /*

     * Inform the source that we're ready to receive a message.

     */

    ret = qemu_rdma_post_send_control(rdma, NULL, &ready);



    if (ret < 0) {

        fprintf(stderr, "Failed to send control buffer!\n");

        return ret;

    }



    /*

     * Block and wait for the message.

     */

    ret = qemu_rdma_exchange_get_response(rdma, head,

                                          expecting, RDMA_WRID_READY);



    if (ret < 0) {

        return ret;

    }



    qemu_rdma_move_header(rdma, RDMA_WRID_READY, head);



    /*

     * Post a new RECV work request to replace the one we just consumed.

     */

    ret = qemu_rdma_post_recv_control(rdma, RDMA_WRID_READY);

    if (ret) {

        fprintf(stderr, "rdma migration: error posting second control recv!");

        return ret;

    }



    return 0;

}
