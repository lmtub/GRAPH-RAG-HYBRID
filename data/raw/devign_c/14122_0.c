static void xen_ram_init(ram_addr_t ram_size)

{

    RAMBlock *new_block;

    ram_addr_t below_4g_mem_size, above_4g_mem_size = 0;



    new_block = qemu_mallocz(sizeof (*new_block));

    pstrcpy(new_block->idstr, sizeof (new_block->idstr), "xen.ram");

    new_block->host = NULL;

    new_block->offset = 0;

    new_block->length = ram_size;



    QLIST_INSERT_HEAD(&ram_list.blocks, new_block, next);



    ram_list.phys_dirty = qemu_realloc(ram_list.phys_dirty,

                                       new_block->length >> TARGET_PAGE_BITS);

    memset(ram_list.phys_dirty + (new_block->offset >> TARGET_PAGE_BITS),

           0xff, new_block->length >> TARGET_PAGE_BITS);



    if (ram_size >= 0xe0000000 ) {

        above_4g_mem_size = ram_size - 0xe0000000;

        below_4g_mem_size = 0xe0000000;

    } else {

        below_4g_mem_size = ram_size;

    }



    cpu_register_physical_memory(0, below_4g_mem_size, new_block->offset);

#if TARGET_PHYS_ADDR_BITS > 32

    if (above_4g_mem_size > 0) {

        cpu_register_physical_memory(0x100000000ULL, above_4g_mem_size,

                                     new_block->offset + below_4g_mem_size);

    }

#endif

}
