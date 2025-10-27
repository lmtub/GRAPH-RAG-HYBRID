static int qemu_rdma_block_for_wrid(RDMAContext *rdma, int wrid_requested)

{

    int num_cq_events = 0, ret = 0;

    struct ibv_cq *cq;

    void *cq_ctx;

    uint64_t wr_id = RDMA_WRID_NONE, wr_id_in;



    if (ibv_req_notify_cq(rdma->cq, 0)) {

        return -1;

    }

    /* poll cq first */

    while (wr_id != wrid_requested) {

        ret = qemu_rdma_poll(rdma, &wr_id_in);

        if (ret < 0) {

            return ret;

        }



        wr_id = wr_id_in & RDMA_WRID_TYPE_MASK;



        if (wr_id == RDMA_WRID_NONE) {

            break;

        }

        if (wr_id != wrid_requested) {

            DDDPRINTF("A Wanted wrid %s (%d) but got %s (%" PRIu64 ")\n",

                print_wrid(wrid_requested),

                wrid_requested, print_wrid(wr_id), wr_id);

        }

    }



    if (wr_id == wrid_requested) {

        return 0;

    }



    while (1) {

        /*

         * Coroutine doesn't start until process_incoming_migration()

         * so don't yield unless we know we're running inside of a coroutine.

         */

        if (rdma->migration_started_on_destination) {

            yield_until_fd_readable(rdma->comp_channel->fd);

        }



        if (ibv_get_cq_event(rdma->comp_channel, &cq, &cq_ctx)) {

            perror("ibv_get_cq_event");

            goto err_block_for_wrid;

        }



        num_cq_events++;



        if (ibv_req_notify_cq(cq, 0)) {

            goto err_block_for_wrid;

        }



        while (wr_id != wrid_requested) {

            ret = qemu_rdma_poll(rdma, &wr_id_in);

            if (ret < 0) {

                goto err_block_for_wrid;

            }



            wr_id = wr_id_in & RDMA_WRID_TYPE_MASK;



            if (wr_id == RDMA_WRID_NONE) {

                break;

            }

            if (wr_id != wrid_requested) {

                DDDPRINTF("B Wanted wrid %s (%d) but got %s (%" PRIu64 ")\n",

                    print_wrid(wrid_requested), wrid_requested,

                    print_wrid(wr_id), wr_id);

            }

        }



        if (wr_id == wrid_requested) {

            goto success_block_for_wrid;

        }

    }



success_block_for_wrid:

    if (num_cq_events) {

        ibv_ack_cq_events(cq, num_cq_events);

    }

    return 0;



err_block_for_wrid:

    if (num_cq_events) {

        ibv_ack_cq_events(cq, num_cq_events);

    }

    return ret;

}
