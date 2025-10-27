static inline TranslationBlock *tb_find_fast(CPUState *cpu,

                                             TranslationBlock *last_tb,

                                             int tb_exit)

{

    CPUArchState *env = (CPUArchState *)cpu->env_ptr;

    TranslationBlock *tb;

    target_ulong cs_base, pc;

    uint32_t flags;



    /* we record a subset of the CPU state. It will

       always be the same before a given translated block

       is executed. */

    cpu_get_tb_cpu_state(env, &pc, &cs_base, &flags);

    tb_lock();

    tb = atomic_rcu_read(&cpu->tb_jmp_cache[tb_jmp_cache_hash_func(pc)]);

    if (unlikely(!tb || tb->pc != pc || tb->cs_base != cs_base ||

                 tb->flags != flags)) {

        tb = tb_find_slow(cpu, pc, cs_base, flags);

    }

#ifndef CONFIG_USER_ONLY

    /* We don't take care of direct jumps when address mapping changes in

     * system emulation. So it's not safe to make a direct jump to a TB

     * spanning two pages because the mapping for the second page can change.

     */

    if (tb->page_addr[1] != -1) {

        last_tb = NULL;

    }

#endif

    /* See if we can patch the calling TB. */

    if (last_tb && !qemu_loglevel_mask(CPU_LOG_TB_NOCHAIN)) {

        /* Check if translation buffer has been flushed */

        if (cpu->tb_flushed) {

            cpu->tb_flushed = false;

        } else {

            tb_add_jump(last_tb, tb_exit, tb);

        }

    }

    tb_unlock();

    return tb;

}
