int cpu_breakpoint_remove(CPUState *cpu, vaddr pc, int flags)

{

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp;



    QTAILQ_FOREACH(bp, &cpu->breakpoints, entry) {

        if (bp->pc == pc && bp->flags == flags) {

            cpu_breakpoint_remove_by_ref(cpu, bp);

            return 0;

        }

    }

    return -ENOENT;

#else

    return -ENOSYS;

#endif

}
