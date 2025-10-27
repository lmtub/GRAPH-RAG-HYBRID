static int qemu_rdma_write_one(QEMUFile *f, RDMAContext *rdma,

                               int current_index, uint64_t current_addr,

                               uint64_t length)

{

    struct ibv_sge sge;

    struct ibv_send_wr send_wr = { 0 };

    struct ibv_send_wr *bad_wr;

    int reg_result_idx, ret, count = 0;

    uint64_t chunk, chunks;

    uint8_t *chunk_start, *chunk_end;

    RDMALocalBlock *block = &(rdma->local_ram_blocks.block[current_index]);

    RDMARegister reg;

    RDMARegisterResult *reg_result;

    RDMAControlHeader resp = { .type = RDMA_CONTROL_REGISTER_RESULT };

    RDMAControlHeader head = { .len = sizeof(RDMARegister),

                               .type = RDMA_CONTROL_REGISTER_REQUEST,

                               .repeat = 1,

                             };



retry:

    sge.addr = (uint64_t)(block->local_host_addr +

                            (current_addr - block->offset));

    sge.length = length;



    chunk = ram_chunk_index(block->local_host_addr, (uint8_t *) sge.addr);

    chunk_start = ram_chunk_start(block, chunk);



    if (block->is_ram_block) {

        chunks = length / (1UL << RDMA_REG_CHUNK_SHIFT);



        if (chunks && ((length % (1UL << RDMA_REG_CHUNK_SHIFT)) == 0)) {

            chunks--;

        }

    } else {

        chunks = block->length / (1UL << RDMA_REG_CHUNK_SHIFT);



        if (chunks && ((block->length % (1UL << RDMA_REG_CHUNK_SHIFT)) == 0)) {

            chunks--;

        }

    }



    DDPRINTF("Writing %" PRIu64 " chunks, (%" PRIu64 " MB)\n",

        chunks + 1, (chunks + 1) * (1UL << RDMA_REG_CHUNK_SHIFT) / 1024 / 1024);



    chunk_end = ram_chunk_end(block, chunk + chunks);



    if (!rdma->pin_all) {

#ifdef RDMA_UNREGISTRATION_EXAMPLE

        qemu_rdma_unregister_waiting(rdma);

#endif

    }



    while (test_bit(chunk, block->transit_bitmap)) {

        (void)count;

        DDPRINTF("(%d) Not clobbering: block: %d chunk %" PRIu64

                " current %" PRIu64 " len %" PRIu64 " %d %d\n",

                count++, current_index, chunk,

                sge.addr, length, rdma->nb_sent, block->nb_chunks);



        ret = qemu_rdma_block_for_wrid(rdma, RDMA_WRID_RDMA_WRITE, NULL);



        if (ret < 0) {

            fprintf(stderr, "Failed to Wait for previous write to complete "

                    "block %d chunk %" PRIu64

                    " current %" PRIu64 " len %" PRIu64 " %d\n",

                    current_index, chunk, sge.addr, length, rdma->nb_sent);

            return ret;

        }

    }



    if (!rdma->pin_all || !block->is_ram_block) {

        if (!block->remote_keys[chunk]) {

            /*

             * This chunk has not yet been registered, so first check to see

             * if the entire chunk is zero. If so, tell the other size to

             * memset() + madvise() the entire chunk without RDMA.

             */



            if (can_use_buffer_find_nonzero_offset((void *)sge.addr, length)

                   && buffer_find_nonzero_offset((void *)sge.addr,

                                                    length) == length) {

                RDMACompress comp = {

                                        .offset = current_addr,

                                        .value = 0,

                                        .block_idx = current_index,

                                        .length = length,

                                    };



                head.len = sizeof(comp);

                head.type = RDMA_CONTROL_COMPRESS;



                DDPRINTF("Entire chunk is zero, sending compress: %"

                    PRIu64 " for %d "

                    "bytes, index: %d, offset: %" PRId64 "...\n",

                    chunk, sge.length, current_index, current_addr);



                compress_to_network(&comp);

                ret = qemu_rdma_exchange_send(rdma, &head,

                                (uint8_t *) &comp, NULL, NULL, NULL);



                if (ret < 0) {

                    return -EIO;

                }



                acct_update_position(f, sge.length, true);



                return 1;

            }



            /*

             * Otherwise, tell other side to register.

             */

            reg.current_index = current_index;

            if (block->is_ram_block) {

                reg.key.current_addr = current_addr;

            } else {

                reg.key.chunk = chunk;

            }

            reg.chunks = chunks;



            DDPRINTF("Sending registration request chunk %" PRIu64 " for %d "

                    "bytes, index: %d, offset: %" PRId64 "...\n",

                    chunk, sge.length, current_index, current_addr);



            register_to_network(&reg);

            ret = qemu_rdma_exchange_send(rdma, &head, (uint8_t *) &reg,

                                    &resp, &reg_result_idx, NULL);

            if (ret < 0) {

                return ret;

            }



            /* try to overlap this single registration with the one we sent. */

            if (qemu_rdma_register_and_get_keys(rdma, block,

                                                (uint8_t *) sge.addr,

                                                &sge.lkey, NULL, chunk,

                                                chunk_start, chunk_end)) {

                fprintf(stderr, "cannot get lkey!\n");

                return -EINVAL;

            }



            reg_result = (RDMARegisterResult *)

                    rdma->wr_data[reg_result_idx].control_curr;



            network_to_result(reg_result);



            DDPRINTF("Received registration result:"

                    " my key: %x their key %x, chunk %" PRIu64 "\n",

                    block->remote_keys[chunk], reg_result->rkey, chunk);



            block->remote_keys[chunk] = reg_result->rkey;

            block->remote_host_addr = reg_result->host_addr;

        } else {

            /* already registered before */

            if (qemu_rdma_register_and_get_keys(rdma, block,

                                                (uint8_t *)sge.addr,

                                                &sge.lkey, NULL, chunk,

                                                chunk_start, chunk_end)) {

                fprintf(stderr, "cannot get lkey!\n");

                return -EINVAL;

            }

        }



        send_wr.wr.rdma.rkey = block->remote_keys[chunk];

    } else {

        send_wr.wr.rdma.rkey = block->remote_rkey;



        if (qemu_rdma_register_and_get_keys(rdma, block, (uint8_t *)sge.addr,

                                                     &sge.lkey, NULL, chunk,

                                                     chunk_start, chunk_end)) {

            fprintf(stderr, "cannot get lkey!\n");

            return -EINVAL;

        }

    }



    /*

     * Encode the ram block index and chunk within this wrid.

     * We will use this information at the time of completion

     * to figure out which bitmap to check against and then which

     * chunk in the bitmap to look for.

     */

    send_wr.wr_id = qemu_rdma_make_wrid(RDMA_WRID_RDMA_WRITE,

                                        current_index, chunk);



    send_wr.opcode = IBV_WR_RDMA_WRITE;

    send_wr.send_flags = IBV_SEND_SIGNALED;

    send_wr.sg_list = &sge;

    send_wr.num_sge = 1;

    send_wr.wr.rdma.remote_addr = block->remote_host_addr +

                                (current_addr - block->offset);



    DDDPRINTF("Posting chunk: %" PRIu64 ", addr: %lx"

              " remote: %lx, bytes %" PRIu32 "\n",

              chunk, sge.addr, send_wr.wr.rdma.remote_addr,

              sge.length);



    /*

     * ibv_post_send() does not return negative error numbers,

     * per the specification they are positive - no idea why.

     */

    ret = ibv_post_send(rdma->qp, &send_wr, &bad_wr);



    if (ret == ENOMEM) {

        DDPRINTF("send queue is full. wait a little....\n");

        ret = qemu_rdma_block_for_wrid(rdma, RDMA_WRID_RDMA_WRITE, NULL);

        if (ret < 0) {

            fprintf(stderr, "rdma migration: failed to make "

                            "room in full send queue! %d\n", ret);

            return ret;

        }



        goto retry;



    } else if (ret > 0) {

        perror("rdma migration: post rdma write failed");

        return -ret;

    }



    set_bit(chunk, block->transit_bitmap);

    acct_update_position(f, sge.length, false);

    rdma->total_writes++;



    return 0;

}
