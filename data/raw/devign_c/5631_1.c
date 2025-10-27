static void update_pam(PCII440FXState *d, uint32_t start, uint32_t end, int r,

                       PAMMemoryRegion *mem)

{

    if (mem->initialized) {

        memory_region_del_subregion(d->system_memory, &mem->mem);

        memory_region_destroy(&mem->mem);

    }



    //    printf("ISA mapping %08x-0x%08x: %d\n", start, end, r);

    switch(r) {

    case 3:

        /* RAM */

        memory_region_init_alias(&mem->mem, "pam-ram", d->ram_memory,

                                 start, end - start);

        break;

    case 1:

        /* ROM (XXX: not quite correct) */

        memory_region_init_alias(&mem->mem, "pam-rom", d->ram_memory,

                                 start, end - start);

        memory_region_set_readonly(&mem->mem, true);

        break;

    case 2:

    case 0:

        /* XXX: should distinguish read/write cases */

        memory_region_init_alias(&mem->mem, "pam-pci", d->pci_address_space,

                                 start, end - start);

        break;

    }

    memory_region_add_subregion_overlap(d->system_memory,

                                        start, &mem->mem, 1);

    mem->initialized = true;

}
