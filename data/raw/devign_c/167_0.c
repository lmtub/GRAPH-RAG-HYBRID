ram_addr_t qemu_ram_alloc_from_ptr(ram_addr_t size, void *host,

                                   MemoryRegion *mr, Error **errp)

{

    RAMBlock *new_block;

    ram_addr_t addr;

    Error *local_err = NULL;



    size = TARGET_PAGE_ALIGN(size);

    new_block = g_malloc0(sizeof(*new_block));

    new_block->mr = mr;

    new_block->used_length = size;

    new_block->max_length = max_size;

    new_block->fd = -1;

    new_block->host = host;

    if (host) {

        new_block->flags |= RAM_PREALLOC;

    }

    addr = ram_block_add(new_block, &local_err);

    if (local_err) {

        g_free(new_block);

        error_propagate(errp, local_err);

        return -1;

    }

    return addr;

}
