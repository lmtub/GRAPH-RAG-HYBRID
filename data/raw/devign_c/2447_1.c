static int qemu_rdma_post_recv_control(RDMAContext *rdma, int idx)

{

    struct ibv_recv_wr *bad_wr;

    struct ibv_sge sge = {

                            .addr = (uint64_t)(rdma->wr_data[idx].control),

                            .length = RDMA_CONTROL_MAX_BUFFER,

                            .lkey = rdma->wr_data[idx].control_mr->lkey,

                         };



    struct ibv_recv_wr recv_wr = {

                                    .wr_id = RDMA_WRID_RECV_CONTROL + idx,

                                    .sg_list = &sge,

                                    .num_sge = 1,

                                 };





    if (ibv_post_recv(rdma->qp, &recv_wr, &bad_wr)) {

        return -1;

    }



    return 0;

}
