static void check_watchpoint(int offset, int len_mask, int flags)

{

    CPUArchState *env = cpu_single_env;

    target_ulong pc, cs_base;

    TranslationBlock *tb;

    target_ulong vaddr;

    CPUWatchpoint *wp;

    int cpu_flags;



    if (env->watchpoint_hit) {

        /* We re-entered the check after replacing the TB. Now raise

         * the debug interrupt so that is will trigger after the

         * current instruction. */

        cpu_interrupt(env, CPU_INTERRUPT_DEBUG);

        return;

    }

    vaddr = (env->mem_io_vaddr & TARGET_PAGE_MASK) + offset;

    QTAILQ_FOREACH(wp, &env->watchpoints, entry) {

        if ((vaddr == (wp->vaddr & len_mask) ||

             (vaddr & wp->len_mask) == wp->vaddr) && (wp->flags & flags)) {

            wp->flags |= BP_WATCHPOINT_HIT;

            if (!env->watchpoint_hit) {

                env->watchpoint_hit = wp;

                tb = tb_find_pc(env->mem_io_pc);

                if (!tb) {

                    cpu_abort(env, "check_watchpoint: could not find TB for "

                              "pc=%p", (void *)env->mem_io_pc);

                }

                cpu_restore_state(tb, env, env->mem_io_pc);

                tb_phys_invalidate(tb, -1);

                if (wp->flags & BP_STOP_BEFORE_ACCESS) {

                    env->exception_index = EXCP_DEBUG;

                    cpu_loop_exit(env);

                } else {

                    cpu_get_tb_cpu_state(env, &pc, &cs_base, &cpu_flags);

                    tb_gen_code(env, pc, cs_base, cpu_flags, 1);

                    cpu_resume_from_signal(env, NULL);

                }

            }

        } else {

            wp->flags &= ~BP_WATCHPOINT_HIT;

        }

    }

}
