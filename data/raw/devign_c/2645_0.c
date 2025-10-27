void *qemu_get_ram_ptr(ram_addr_t addr)

{

    RAMBlock *prev;

    RAMBlock **prevp;

    RAMBlock *block;



#ifdef CONFIG_KQEMU

    if (kqemu_phys_ram_base) {

        return kqemu_phys_ram_base + addr;

    }

#endif



    prev = NULL;

    prevp = &ram_blocks;

    block = ram_blocks;

    while (block && (block->offset > addr

                     || block->offset + block->length <= addr)) {

        if (prev)

          prevp = &prev->next;

        prev = block;

        block = block->next;

    }

    if (!block) {

        fprintf(stderr, "Bad ram offset %" PRIx64 "\n", (uint64_t)addr);

        abort();

    }

    /* Move this entry to to start of the list.  */

    if (prev) {

        prev->next = block->next;

        block->next = *prevp;

        *prevp = block;

    }

    return block->host + (addr - block->offset);

}
