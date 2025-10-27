static void qemu_rdma_cleanup(RDMAContext *rdma)

{

    struct rdma_cm_event *cm_event;

    int ret, idx;



    if (rdma->cm_id && rdma->connected) {

        if (rdma->error_state) {

            RDMAControlHeader head = { .len = 0,

                                       .type = RDMA_CONTROL_ERROR,

                                       .repeat = 1,

                                     };

            error_report("Early error. Sending error.");

            qemu_rdma_post_send_control(rdma, NULL, &head);

        }



        ret = rdma_disconnect(rdma->cm_id);

        if (!ret) {

            trace_qemu_rdma_cleanup_waiting_for_disconnect();

            ret = rdma_get_cm_event(rdma->channel, &cm_event);

            if (!ret) {

                rdma_ack_cm_event(cm_event);

            }

        }

        trace_qemu_rdma_cleanup_disconnect();

        rdma->connected = false;

    }



    g_free(rdma->block);

    rdma->block = NULL;



    for (idx = 0; idx < RDMA_WRID_MAX; idx++) {

        if (rdma->wr_data[idx].control_mr) {

            rdma->total_registrations--;

            ibv_dereg_mr(rdma->wr_data[idx].control_mr);

        }

        rdma->wr_data[idx].control_mr = NULL;

    }



    if (rdma->local_ram_blocks.block) {

        while (rdma->local_ram_blocks.nb_blocks) {

            rdma_delete_block(rdma, rdma->local_ram_blocks.block->offset);

        }

    }



    if (rdma->cq) {

        ibv_destroy_cq(rdma->cq);

        rdma->cq = NULL;

    }

    if (rdma->comp_channel) {

        ibv_destroy_comp_channel(rdma->comp_channel);

        rdma->comp_channel = NULL;

    }

    if (rdma->pd) {

        ibv_dealloc_pd(rdma->pd);

        rdma->pd = NULL;

    }

    if (rdma->listen_id) {

        rdma_destroy_id(rdma->listen_id);

        rdma->listen_id = NULL;

    }

    if (rdma->cm_id) {

        if (rdma->qp) {

            rdma_destroy_qp(rdma->cm_id);

            rdma->qp = NULL;

        }

        rdma_destroy_id(rdma->cm_id);

        rdma->cm_id = NULL;

    }

    if (rdma->channel) {

        rdma_destroy_event_channel(rdma->channel);

        rdma->channel = NULL;

    }

    g_free(rdma->host);

    rdma->host = NULL;

}
