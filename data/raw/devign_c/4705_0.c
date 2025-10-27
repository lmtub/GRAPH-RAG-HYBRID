static int tcg_cpu_exec(CPUState *cpu)

{

    int ret;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif



#ifdef CONFIG_PROFILER

    ti = profile_getclock();

#endif

    if (use_icount) {

        int64_t count;

        int64_t deadline;

        int decr;

        timers_state.qemu_icount -= (cpu->icount_decr.u16.low

                                    + cpu->icount_extra);

        cpu->icount_decr.u16.low = 0;

        cpu->icount_extra = 0;

        deadline = qemu_clock_deadline_ns_all(QEMU_CLOCK_VIRTUAL);



        /* Maintain prior (possibly buggy) behaviour where if no deadline

         * was set (as there is no QEMU_CLOCK_VIRTUAL timer) or it is more than

         * INT32_MAX nanoseconds ahead, we still use INT32_MAX

         * nanoseconds.

         */

        if ((deadline < 0) || (deadline > INT32_MAX)) {

            deadline = INT32_MAX;

        }



        count = qemu_icount_round(deadline);

        timers_state.qemu_icount += count;

        decr = (count > 0xffff) ? 0xffff : count;

        count -= decr;

        cpu->icount_decr.u16.low = decr;

        cpu->icount_extra = count;

    }

    ret = cpu_exec(cpu);

#ifdef CONFIG_PROFILER

    tcg_time += profile_getclock() - ti;

#endif

    if (use_icount) {

        /* Fold pending instructions back into the

           instruction counter, and clear the interrupt flag.  */

        timers_state.qemu_icount -= (cpu->icount_decr.u16.low

                        + cpu->icount_extra);

        cpu->icount_decr.u32 = 0;

        cpu->icount_extra = 0;

    }

    return ret;

}
