int qemu_ram_addr_from_host(void *ptr, ram_addr_t *ram_addr)

{

    RAMBlock *block;

    uint8_t *host = ptr;



    if (xen_enabled()) {

        *ram_addr = xen_ram_addr_from_mapcache(ptr);

        return 0;

    }



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        /* This case append when the block is not mapped. */

        if (block->host == NULL) {

            continue;

        }

        if (host - block->host < block->length) {

            *ram_addr = block->offset + (host - block->host);

            return 0;

        }

    }



    return -1;

}
