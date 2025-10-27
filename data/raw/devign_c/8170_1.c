static int qemu_rdma_registration_handle(QEMUFile *f, void *opaque)

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

        trace_qemu_rdma_registration_handle_wait();



        ret = qemu_rdma_exchange_recv(rdma, &head, RDMA_CONTROL_NONE);



        if (ret < 0) {





        if (head.repeat > RDMA_CONTROL_MAX_COMMANDS_PER_MESSAGE) {

            error_report("rdma: Too many requests in this message (%d)."

                            "Bailing.", head.repeat);

            ret = -EIO;





        switch (head.type) {

        case RDMA_CONTROL_COMPRESS:

            comp = (RDMACompress *) rdma->wr_data[idx].control_curr;

            network_to_compress(comp);



            trace_qemu_rdma_registration_handle_compress(comp->length,

                                                         comp->block_idx,

                                                         comp->offset);

            if (comp->block_idx >= rdma->local_ram_blocks.nb_blocks) {

                error_report("rdma: 'compress' bad block index %u (vs %d)",

                             (unsigned int)comp->block_idx,

                             rdma->local_ram_blocks.nb_blocks);

                ret = -EIO;



            block = &(rdma->local_ram_blocks.block[comp->block_idx]);



            host_addr = block->local_host_addr +

                            (comp->offset - block->offset);



            ram_handle_compressed(host_addr, comp->value, comp->length);




        case RDMA_CONTROL_REGISTER_FINISHED:

            trace_qemu_rdma_registration_handle_finished();

            goto out;



        case RDMA_CONTROL_RAM_BLOCKS_REQUEST:

            trace_qemu_rdma_registration_handle_ram_blocks();



            /* Sort our local RAM Block list so it's the same as the source,

             * we can do this since we've filled in a src_index in the list

             * as we received the RAMBlock list earlier.

             */

            qsort(rdma->local_ram_blocks.block,

                  rdma->local_ram_blocks.nb_blocks,

                  sizeof(RDMALocalBlock), dest_ram_sort_func);

            if (rdma->pin_all) {

                ret = qemu_rdma_reg_whole_ram_blocks(rdma);

                if (ret) {

                    error_report("rdma migration: error dest "

                                    "registering ram blocks");

                    goto out;





            /*

             * Dest uses this to prepare to transmit the RAMBlock descriptions

             * to the source VM after connection setup.

             * Both sides use the "remote" structure to communicate and update

             * their "local" descriptions with what was sent.

             */

            for (i = 0; i < local->nb_blocks; i++) {

                rdma->dest_blocks[i].remote_host_addr =

                    (uintptr_t)(local->block[i].local_host_addr);



                if (rdma->pin_all) {

                    rdma->dest_blocks[i].remote_rkey = local->block[i].mr->rkey;




                rdma->dest_blocks[i].offset = local->block[i].offset;

                rdma->dest_blocks[i].length = local->block[i].length;



                dest_block_to_network(&rdma->dest_blocks[i]);

                trace_qemu_rdma_registration_handle_ram_blocks_loop(

                    local->block[i].block_name,

                    local->block[i].offset,

                    local->block[i].length,

                    local->block[i].local_host_addr,

                    local->block[i].src_index);




            blocks.len = rdma->local_ram_blocks.nb_blocks

                                                * sizeof(RDMADestBlock);





            ret = qemu_rdma_post_send_control(rdma,

                                        (uint8_t *) rdma->dest_blocks, &blocks);



            if (ret < 0) {

                error_report("rdma migration: error sending remote info");

                goto out;





        case RDMA_CONTROL_REGISTER_REQUEST:

            trace_qemu_rdma_registration_handle_register(head.repeat);



            reg_resp.repeat = head.repeat;

            registers = (RDMARegister *) rdma->wr_data[idx].control_curr;



            for (count = 0; count < head.repeat; count++) {

                uint64_t chunk;

                uint8_t *chunk_start, *chunk_end;



                reg = &registers[count];

                network_to_register(reg);



                reg_result = &results[count];



                trace_qemu_rdma_registration_handle_register_loop(count,

                         reg->current_index, reg->key.current_addr, reg->chunks);



                if (reg->current_index >= rdma->local_ram_blocks.nb_blocks) {

                    error_report("rdma: 'register' bad block index %u (vs %d)",

                                 (unsigned int)reg->current_index,

                                 rdma->local_ram_blocks.nb_blocks);

                    ret = -ENOENT;



                block = &(rdma->local_ram_blocks.block[reg->current_index]);

                if (block->is_ram_block) {

                    if (block->offset > reg->key.current_addr) {

                        error_report("rdma: bad register address for block %s"

                            " offset: %" PRIx64 " current_addr: %" PRIx64,

                            block->block_name, block->offset,

                            reg->key.current_addr);




                    host_addr = (block->local_host_addr +

                                (reg->key.current_addr - block->offset));

                    chunk = ram_chunk_index(block->local_host_addr,

                                            (uint8_t *) host_addr);

                } else {

                    chunk = reg->key.chunk;

                    host_addr = block->local_host_addr +

                        (reg->key.chunk * (1UL << RDMA_REG_CHUNK_SHIFT));










                chunk_start = ram_chunk_start(block, chunk);

                chunk_end = ram_chunk_end(block, chunk + reg->chunks);

                if (qemu_rdma_register_and_get_keys(rdma, block,

                            (uintptr_t)host_addr, NULL, &reg_result->rkey,

                            chunk, chunk_start, chunk_end)) {

                    error_report("cannot get rkey");

                    ret = -EINVAL;

                    goto out;




                reg_result->host_addr = (uintptr_t)block->local_host_addr;



                trace_qemu_rdma_registration_handle_register_rkey(

                                                           reg_result->rkey);



                result_to_network(reg_result);




            ret = qemu_rdma_post_send_control(rdma,

                            (uint8_t *) results, &reg_resp);



            if (ret < 0) {

                error_report("Failed to send control buffer");

                goto out;



        case RDMA_CONTROL_UNREGISTER_REQUEST:

            trace_qemu_rdma_registration_handle_unregister(head.repeat);

            unreg_resp.repeat = head.repeat;

            registers = (RDMARegister *) rdma->wr_data[idx].control_curr;



            for (count = 0; count < head.repeat; count++) {

                reg = &registers[count];

                network_to_register(reg);



                trace_qemu_rdma_registration_handle_unregister_loop(count,

                           reg->current_index, reg->key.chunk);



                block = &(rdma->local_ram_blocks.block[reg->current_index]);



                ret = ibv_dereg_mr(block->pmr[reg->key.chunk]);

                block->pmr[reg->key.chunk] = NULL;



                if (ret != 0) {

                    perror("rdma unregistration chunk failed");

                    ret = -ret;

                    goto out;




                rdma->total_registrations--;



                trace_qemu_rdma_registration_handle_unregister_success(

                                                       reg->key.chunk);




            ret = qemu_rdma_post_send_control(rdma, NULL, &unreg_resp);



            if (ret < 0) {

                error_report("Failed to send control buffer");

                goto out;



        case RDMA_CONTROL_REGISTER_RESULT:

            error_report("Invalid RESULT message at dest.");

            ret = -EIO;

            goto out;

        default:

            error_report("Unknown control message %s", control_desc[head.type]);

            ret = -EIO;

            goto out;


    } while (1);

out:

    if (ret < 0) {

        rdma->error_state = ret;


    return ret;
