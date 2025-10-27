static void notdirty_mem_writeb(void *opaque, target_phys_addr_t ram_addr,

                                uint32_t val)

{

    int dirty_flags;

    dirty_flags = phys_ram_dirty[ram_addr >> TARGET_PAGE_BITS];

    if (!(dirty_flags & CODE_DIRTY_FLAG)) {

#if !defined(CONFIG_USER_ONLY)

        tb_invalidate_phys_page_fast(ram_addr, 1);

        dirty_flags = phys_ram_dirty[ram_addr >> TARGET_PAGE_BITS];

#endif

    }

    stb_p(qemu_get_ram_ptr(ram_addr), val);

#ifdef CONFIG_KQEMU

    if (cpu_single_env->kqemu_enabled &&

        (dirty_flags & KQEMU_MODIFY_PAGE_MASK) != KQEMU_MODIFY_PAGE_MASK)

        kqemu_modify_page(cpu_single_env, ram_addr);

#endif

    dirty_flags |= (0xff & ~CODE_DIRTY_FLAG);

    phys_ram_dirty[ram_addr >> TARGET_PAGE_BITS] = dirty_flags;

    /* we remove the notdirty callback only if the code has been

       flushed */

    if (dirty_flags == 0xff)

        tlb_set_dirty(cpu_single_env, cpu_single_env->mem_io_vaddr);

}
