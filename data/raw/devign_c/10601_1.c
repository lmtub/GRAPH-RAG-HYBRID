static int qemu_rdma_init_ram_blocks(RDMAContext *rdma)

{

    RDMALocalBlocks *local = &rdma->local_ram_blocks;



    assert(rdma->blockmap == NULL);

    rdma->blockmap = g_hash_table_new(g_direct_hash, g_direct_equal);

    memset(local, 0, sizeof *local);

    qemu_ram_foreach_block(qemu_rdma_init_one_block, rdma);

    DPRINTF("Allocated %d local ram block structures\n", local->nb_blocks);

    rdma->block = (RDMARemoteBlock *) g_malloc0(sizeof(RDMARemoteBlock) *

                        rdma->local_ram_blocks.nb_blocks);

    local->init = true;

    return 0;

}
