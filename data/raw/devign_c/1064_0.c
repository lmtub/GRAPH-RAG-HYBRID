void kqemu_set_notdirty(CPUState *env, ram_addr_t ram_addr)

{

    LOG_INT("kqemu_set_notdirty: addr=%08lx\n", 

                (unsigned long)ram_addr);

    /* we only track transitions to dirty state */

    if (phys_ram_dirty[ram_addr >> TARGET_PAGE_BITS] != 0xff)

        return;

    if (nb_ram_pages_to_update >= KQEMU_MAX_RAM_PAGES_TO_UPDATE)

        nb_ram_pages_to_update = KQEMU_RAM_PAGES_UPDATE_ALL;

    else

        ram_pages_to_update[nb_ram_pages_to_update++] = ram_addr;

}
