int cpu_watchpoint_insert(CPUState *env, target_ulong addr, target_ulong len,

                          int flags, CPUWatchpoint **watchpoint)

{

    target_ulong len_mask = ~(len - 1);

    CPUWatchpoint *wp;



    /* sanity checks: allow power-of-2 lengths, deny unaligned watchpoints */

    if ((len != 1 && len != 2 && len != 4 && len != 8) || (addr & ~len_mask)) {

        fprintf(stderr, "qemu: tried to set invalid watchpoint at "

                TARGET_FMT_lx ", len=" TARGET_FMT_lu "\n", addr, len);

        return -EINVAL;

    }

    wp = qemu_malloc(sizeof(*wp));



    wp->vaddr = addr;

    wp->len_mask = len_mask;

    wp->flags = flags;



    /* keep all GDB-injected watchpoints in front */

    if (flags & BP_GDB)

        TAILQ_INSERT_HEAD(&env->watchpoints, wp, entry);

    else

        TAILQ_INSERT_TAIL(&env->watchpoints, wp, entry);



    tlb_flush_page(env, addr);



    if (watchpoint)

        *watchpoint = wp;

    return 0;

}
