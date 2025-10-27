void *qemu_ram_ptr_length(target_phys_addr_t addr, target_phys_addr_t *size)

{

    if (xen_enabled()) {

        return xen_map_cache(addr, *size, 1);

    } else {

        RAMBlock *block;



        QLIST_FOREACH(block, &ram_list.blocks, next) {

            if (addr - block->offset < block->length) {

                if (addr - block->offset + *size > block->length)

                    *size = block->length - addr + block->offset;

                return block->host + (addr - block->offset);

            }

        }



        fprintf(stderr, "Bad ram offset %" PRIx64 "\n", (uint64_t)addr);

        abort();



        *size = 0;

        return NULL;

    }

}
