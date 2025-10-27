ram_addr_t qemu_ram_addr_from_host(void *ptr)

{

    RAMBlock *prev;

    RAMBlock **prevp;

    RAMBlock *block;

    uint8_t *host = ptr;



#ifdef CONFIG_KQEMU

    if (kqemu_phys_ram_base) {

        return host - kqemu_phys_ram_base;

    }

#endif



    prev = NULL;

    prevp = &ram_blocks;

    block = ram_blocks;

    while (block && (block->host > host

                     || block->host + block->length <= host)) {

        if (prev)

          prevp = &prev->next;

        prev = block;

        block = block->next;

    }

    if (!block) {

        fprintf(stderr, "Bad ram pointer %p\n", ptr);

        abort();

    }

    return block->offset + (host - block->host);

}
