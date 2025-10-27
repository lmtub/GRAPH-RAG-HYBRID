int cpu_watchpoint_insert(CPUState *cpu, vaddr addr, vaddr len,

                          int flags, CPUWatchpoint **watchpoint)

{

    CPUWatchpoint *wp;



    /* forbid ranges which are empty or run off the end of the address space */

    if (len == 0 || (addr + len - 1) <= addr) {

        error_report("tried to set invalid watchpoint at %"

                     VADDR_PRIx ", len=%" VADDR_PRIu, addr, len);

        return -EINVAL;

    }

    wp = g_malloc(sizeof(*wp));



    wp->vaddr = addr;

    wp->len = len;

    wp->flags = flags;



    /* keep all GDB-injected watchpoints in front */

    if (flags & BP_GDB) {

        QTAILQ_INSERT_HEAD(&cpu->watchpoints, wp, entry);

    } else {

        QTAILQ_INSERT_TAIL(&cpu->watchpoints, wp, entry);

    }



    tlb_flush_page(cpu, addr);



    if (watchpoint)

        *watchpoint = wp;

    return 0;

}
