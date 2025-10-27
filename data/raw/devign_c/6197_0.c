void cpu_watchpoint_remove_by_ref(CPUState *env, CPUWatchpoint *watchpoint)

{

    TAILQ_REMOVE(&env->watchpoints, watchpoint, entry);



    tlb_flush_page(env, watchpoint->vaddr);



    qemu_free(watchpoint);

}
