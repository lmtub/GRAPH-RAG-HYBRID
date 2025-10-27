void cpu_breakpoint_remove_by_ref(CPUState *cpu, CPUBreakpoint *breakpoint)

{

#if defined(TARGET_HAS_ICE)

    QTAILQ_REMOVE(&cpu->breakpoints, breakpoint, entry);



    breakpoint_invalidate(cpu, breakpoint->pc);



    g_free(breakpoint);

#endif

}
