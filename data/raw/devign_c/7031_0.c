static void notdirty_mem_write(void *opaque, target_phys_addr_t ram_addr,

                               uint64_t val, unsigned size)

{

    int dirty_flags;

    dirty_flags = cpu_physical_memory_get_dirty_flags(ram_addr);

    if (!(dirty_flags & CODE_DIRTY_FLAG)) {

#if !defined(CONFIG_USER_ONLY)

        tb_invalidate_phys_page_fast(ram_addr, size);

        dirty_flags = cpu_physical_memory_get_dirty_flags(ram_addr);

#endif

    }

    switch (size) {

    case 1:

        stb_p(qemu_get_ram_ptr(ram_addr), val);

        break;

    case 2:

        stw_p(qemu_get_ram_ptr(ram_addr), val);

        break;

    case 4:

        stl_p(qemu_get_ram_ptr(ram_addr), val);

        break;

    default:

        abort();

    }

    dirty_flags |= (0xff & ~CODE_DIRTY_FLAG);

    cpu_physical_memory_set_dirty_flags(ram_addr, dirty_flags);

    /* we remove the notdirty callback only if the code has been

       flushed */

    if (dirty_flags == 0xff)

        tlb_set_dirty(cpu_single_env, cpu_single_env->mem_io_vaddr);

}
