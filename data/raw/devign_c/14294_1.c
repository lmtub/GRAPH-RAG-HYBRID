static int qemu_rdma_registration_stop(QEMUFile *f, void *opaque,

                                       uint64_t flags)

{

    Error *local_err = NULL, **errp = &local_err;

    QEMUFileRDMA *rfile = opaque;

    RDMAContext *rdma = rfile->rdma;

    RDMAControlHeader head = { .len = 0, .repeat = 1 };

    int ret = 0;



    CHECK_ERROR_STATE();



    qemu_fflush(f);

    ret = qemu_rdma_drain_cq(f, rdma);



    if (ret < 0) {

        goto err;

    }



    if (flags == RAM_CONTROL_SETUP) {

        RDMAControlHeader resp = {.type = RDMA_CONTROL_RAM_BLOCKS_RESULT };

        RDMALocalBlocks *local = &rdma->local_ram_blocks;

        int reg_result_idx, i, j, nb_remote_blocks;



        head.type = RDMA_CONTROL_RAM_BLOCKS_REQUEST;

        DPRINTF("Sending registration setup for ram blocks...\n");



        /*

         * Make sure that we parallelize the pinning on both sides.

         * For very large guests, doing this serially takes a really

         * long time, so we have to 'interleave' the pinning locally

         * with the control messages by performing the pinning on this

         * side before we receive the control response from the other

         * side that the pinning has completed.

         */

        ret = qemu_rdma_exchange_send(rdma, &head, NULL, &resp,

                    &reg_result_idx, rdma->pin_all ?

                    qemu_rdma_reg_whole_ram_blocks : NULL);

        if (ret < 0) {

            ERROR(errp, "receiving remote info!");

            return ret;

        }



        nb_remote_blocks = resp.len / sizeof(RDMARemoteBlock);



        /*

         * The protocol uses two different sets of rkeys (mutually exclusive):

         * 1. One key to represent the virtual address of the entire ram block.

         *    (dynamic chunk registration disabled - pin everything with one rkey.)

         * 2. One to represent individual chunks within a ram block.

         *    (dynamic chunk registration enabled - pin individual chunks.)

         *

         * Once the capability is successfully negotiated, the destination transmits

         * the keys to use (or sends them later) including the virtual addresses

         * and then propagates the remote ram block descriptions to his local copy.

         */



        if (local->nb_blocks != nb_remote_blocks) {

            ERROR(errp, "ram blocks mismatch #1! "

                        "Your QEMU command line parameters are probably "

                        "not identical on both the source and destination.");

            return -EINVAL;

        }



        qemu_rdma_move_header(rdma, reg_result_idx, &resp);

        memcpy(rdma->block,

            rdma->wr_data[reg_result_idx].control_curr, resp.len);

        for (i = 0; i < nb_remote_blocks; i++) {

            network_to_remote_block(&rdma->block[i]);



            /* search local ram blocks */

            for (j = 0; j < local->nb_blocks; j++) {

                if (rdma->block[i].offset != local->block[j].offset) {

                    continue;

                }



                if (rdma->block[i].length != local->block[j].length) {

                    ERROR(errp, "ram blocks mismatch #2! "

                        "Your QEMU command line parameters are probably "

                        "not identical on both the source and destination.");

                    return -EINVAL;

                }

                local->block[j].remote_host_addr =

                        rdma->block[i].remote_host_addr;

                local->block[j].remote_rkey = rdma->block[i].remote_rkey;

                break;

            }



            if (j >= local->nb_blocks) {

                ERROR(errp, "ram blocks mismatch #3! "

                        "Your QEMU command line parameters are probably "

                        "not identical on both the source and destination.");

                return -EINVAL;

            }

        }

    }



    DDDPRINTF("Sending registration finish %" PRIu64 "...\n", flags);



    head.type = RDMA_CONTROL_REGISTER_FINISHED;

    ret = qemu_rdma_exchange_send(rdma, &head, NULL, NULL, NULL, NULL);



    if (ret < 0) {

        goto err;

    }



    return 0;

err:

    rdma->error_state = ret;

    return ret;

}
