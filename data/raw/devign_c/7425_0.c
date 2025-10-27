static void invalidate_and_set_dirty(target_phys_addr_t addr,

                                     target_phys_addr_t length)

{

    if (!cpu_physical_memory_is_dirty(addr)) {

        /* invalidate code */

        tb_invalidate_phys_page_range(addr, addr + length, 0);

        /* set dirty bit */

        cpu_physical_memory_set_dirty_flags(addr, (0xff & ~CODE_DIRTY_FLAG));

    }

    xen_modified_memory(addr, length);

}
