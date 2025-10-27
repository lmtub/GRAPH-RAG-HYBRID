static int qemu_rdma_registration_handle(QEMUFile *f, void *opaque,

                                         uint64_t flags)

{

    RDMAControlHeader reg_resp = { .len = sizeof(RDMARegisterResult),

                               .type = RDMA_CONTROL_REGISTER_RESULT,

                               .repeat = 0,

                             };

    RDMAControlHeader unreg_resp = { .len = 0,

                               .type = RDMA_CONTROL_UNREGISTER_FINISHED,

                               .repeat = 0,

                             };

    RDMAControlHeader blocks = { .type = RDMA_CONTROL_RAM_BLOCKS_RESULT,

                                 .repeat = 1 };

    QEMUFileRDMA *rfile = opaque;

    RDMAContext *rdma = rfile->rdma;

    RDMALocalBlocks *local = &rdma->local_ram_blocks;

    RDMAControlHeader head;

    RDMARegister *reg, *registers;

    RDMACompress *comp;

    RDMARegisterResult *reg_result;

    static RDMARegisterResult results[RDMA_CONTROL_MAX_COMMANDS_PER_MESSAGE];

    RDMALocalBlock *block;

    void *host_addr;

    int ret = 0;

    int idx = 0;

    int count = 0;

    int i = 0;



    CHECK_ERROR_STATE();



    do {

        DDDPRINTF("Waiting for next request %" PRIu64 "...\n", flags);



        ret = qemu_rdma_exchange_recv(rdma, &head, RDMA_CONTROL_NONE);



        if (ret < 0) {

            break;

        }



        if (head.repeat > RDMA_CONTROL_MAX_COMMANDS_PER_MESSAGE) {

            fprintf(stderr, "rdma: Too many requests in this message (%d)."

                            "Bailing.\n", head.repeat);

            ret = -EIO;

            break;

        }



        switch (head.type) {

        case RDMA_CONTROL_COMPRESS:

            comp = (RDMACompress *) rdma->wr_data[idx].control_curr;

            network_to_compress(comp);



            DDPRINTF("Zapping zero chunk: %" PRId64

                    " bytes, index %d, offset %" PRId64 "\n",

                    comp->length, comp->block_idx, comp->offset);

            block = &(rdma->local_ram_blocks.block[comp->block_idx]);



            host_addr = block->local_host_addr +

                            (comp->offset - block->offset);



            ram_handle_compressed(host_addr, comp->value, comp->length);

            break;



        case RDMA_CONTROL_REGISTER_FINISHED:

            DDDPRINTF("Current registrations complete.\n");

            goto out;



        case RDMA_CONTROL_RAM_BLOCKS_REQUEST:

            DPRINTF("Initial setup info requested.\n");



            if (rdma->pin_all) {

                ret = qemu_rdma_reg_whole_ram_blocks(rdma);

                if (ret) {

                    fprintf(stderr, "rdma migration: error dest "

                                    "registering ram blocks!\n");

                    goto out;

                }

            }



            /*

             * Dest uses this to prepare to transmit the RAMBlock descriptions

             * to the source VM after connection setup.

             * Both sides use the "remote" structure to communicate and update

             * their "local" descriptions with what was sent.

             */

            for (i = 0; i < local->nb_blocks; i++) {

                rdma->block[i].remote_host_addr =

                    (uint64_t)(local->block[i].local_host_addr);



                if (rdma->pin_all) {

                    rdma->block[i].remote_rkey = local->block[i].mr->rkey;

                }



                rdma->block[i].offset = local->block[i].offset;

                rdma->block[i].length = local->block[i].length;



                remote_block_to_network(&rdma->block[i]);

            }



            blocks.len = rdma->local_ram_blocks.nb_blocks

                                                * sizeof(RDMARemoteBlock);





            ret = qemu_rdma_post_send_control(rdma,

                                        (uint8_t *) rdma->block, &blocks);



            if (ret < 0) {

                fprintf(stderr, "rdma migration: error sending remote info!\n");

                goto out;

            }



            break;

        case RDMA_CONTROL_REGISTER_REQUEST:

            DDPRINTF("There are %d registration requests\n", head.repeat);



            reg_resp.repeat = head.repeat;

            registers = (RDMARegister *) rdma->wr_data[idx].control_curr;



            for (count = 0; count < head.repeat; count++) {

                uint64_t chunk;

                uint8_t *chunk_start, *chunk_end;



                reg = &registers[count];

                network_to_register(reg);



                reg_result = &results[count];



                DDPRINTF("Registration request (%d): index %d, current_addr %"

                         PRIu64 " chunks: %" PRIu64 "\n", count,

                         reg->current_index, reg->key.current_addr, reg->chunks);



                block = &(rdma->local_ram_blocks.block[reg->current_index]);

                if (block->is_ram_block) {

                    host_addr = (block->local_host_addr +

                                (reg->key.current_addr - block->offset));

                    chunk = ram_chunk_index(block->local_host_addr,

                                            (uint8_t *) host_addr);

                } else {

                    chunk = reg->key.chunk;

                    host_addr = block->local_host_addr +

                        (reg->key.chunk * (1UL << RDMA_REG_CHUNK_SHIFT));

                }

                chunk_start = ram_chunk_start(block, chunk);

                chunk_end = ram_chunk_end(block, chunk + reg->chunks);

                if (qemu_rdma_register_and_get_keys(rdma, block,

                            (uint8_t *)host_addr, NULL, &reg_result->rkey,

                            chunk, chunk_start, chunk_end)) {

                    fprintf(stderr, "cannot get rkey!\n");

                    ret = -EINVAL;

                    goto out;

                }



                reg_result->host_addr = (uint64_t) block->local_host_addr;



                DDPRINTF("Registered rkey for this request: %x\n",

                                reg_result->rkey);



                result_to_network(reg_result);

            }



            ret = qemu_rdma_post_send_control(rdma,

                            (uint8_t *) results, &reg_resp);



            if (ret < 0) {

                fprintf(stderr, "Failed to send control buffer!\n");

                goto out;

            }

            break;

        case RDMA_CONTROL_UNREGISTER_REQUEST:

            DDPRINTF("There are %d unregistration requests\n", head.repeat);

            unreg_resp.repeat = head.repeat;

            registers = (RDMARegister *) rdma->wr_data[idx].control_curr;



            for (count = 0; count < head.repeat; count++) {

                reg = &registers[count];

                network_to_register(reg);



                DDPRINTF("Unregistration request (%d): "

                         " index %d, chunk %" PRIu64 "\n",

                         count, reg->current_index, reg->key.chunk);



                block = &(rdma->local_ram_blocks.block[reg->current_index]);



                ret = ibv_dereg_mr(block->pmr[reg->key.chunk]);

                block->pmr[reg->key.chunk] = NULL;



                if (ret != 0) {

                    perror("rdma unregistration chunk failed");

                    ret = -ret;

                    goto out;

                }



                rdma->total_registrations--;



                DDPRINTF("Unregistered chunk %" PRIu64 " successfully.\n",

                            reg->key.chunk);

            }



            ret = qemu_rdma_post_send_control(rdma, NULL, &unreg_resp);



            if (ret < 0) {

                fprintf(stderr, "Failed to send control buffer!\n");

                goto out;

            }

            break;

        case RDMA_CONTROL_REGISTER_RESULT:

            fprintf(stderr, "Invalid RESULT message at dest.\n");

            ret = -EIO;

            goto out;

        default:

            fprintf(stderr, "Unknown control message %s\n",

                                control_desc[head.type]);

            ret = -EIO;

            goto out;

        }

    } while (1);

out:

    if (ret < 0) {

        rdma->error_state = ret;

    }

    return ret;

}
