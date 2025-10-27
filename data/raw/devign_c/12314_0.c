static RAMBlock *qemu_get_ram_block(ram_addr_t addr)

{

    RAMBlock *block;



    /* The list is protected by the iothread lock here.  */

    block = ram_list.mru_block;

    if (block && addr - block->offset < block->max_length) {

        goto found;

    }

    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (addr - block->offset < block->max_length) {

            goto found;

        }

    }



    fprintf(stderr, "Bad ram offset %" PRIx64 "\n", (uint64_t)addr);

    abort();



found:

    ram_list.mru_block = block;

    return block;

}
