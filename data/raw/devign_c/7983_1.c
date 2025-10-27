static int main_loop(void)

{

    int ret, timeout;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif

    CPUState *env;



    cur_cpu = first_cpu;

    next_cpu = cur_cpu->next_cpu ?: first_cpu;

    for(;;) {

        if (vm_running) {



            for(;;) {

                /* get next cpu */

                env = next_cpu;

#ifdef CONFIG_PROFILER

                ti = profile_getclock();

#endif

                if (use_icount) {

                    int64_t count;

                    int decr;

                    qemu_icount -= (env->icount_decr.u16.low + env->icount_extra);

                    env->icount_decr.u16.low = 0;

                    env->icount_extra = 0;

                    count = qemu_next_deadline();

                    count = (count + (1 << icount_time_shift) - 1)

                            >> icount_time_shift;

                    qemu_icount += count;

                    decr = (count > 0xffff) ? 0xffff : count;

                    count -= decr;

                    env->icount_decr.u16.low = decr;

                    env->icount_extra = count;

                }

                ret = cpu_exec(env);

#ifdef CONFIG_PROFILER

                qemu_time += profile_getclock() - ti;

#endif

                if (use_icount) {

                    /* Fold pending instructions back into the

                       instruction counter, and clear the interrupt flag.  */

                    qemu_icount -= (env->icount_decr.u16.low

                                    + env->icount_extra);

                    env->icount_decr.u32 = 0;

                    env->icount_extra = 0;

                }

                next_cpu = env->next_cpu ?: first_cpu;

                if (event_pending && likely(ret != EXCP_DEBUG)) {

                    ret = EXCP_INTERRUPT;

                    event_pending = 0;

                    break;

                }

                if (ret == EXCP_HLT) {

                    /* Give the next CPU a chance to run.  */

                    cur_cpu = env;

                    continue;

                }

                if (ret != EXCP_HALTED)

                    break;

                /* all CPUs are halted ? */

                if (env == cur_cpu)

                    break;

            }

            cur_cpu = env;



            if (shutdown_requested) {

                ret = EXCP_INTERRUPT;

                if (no_shutdown) {

                    vm_stop(0);

                    no_shutdown = 0;

                }

                else

                    break;

            }

            if (reset_requested) {

                reset_requested = 0;

                qemu_system_reset();

                ret = EXCP_INTERRUPT;

            }

            if (powerdown_requested) {

                powerdown_requested = 0;

		qemu_system_powerdown();

                ret = EXCP_INTERRUPT;

            }

            if (unlikely(ret == EXCP_DEBUG)) {

                vm_stop(EXCP_DEBUG);

            }

            /* If all cpus are halted then wait until the next IRQ */

            /* XXX: use timeout computed from timers */

            if (ret == EXCP_HALTED) {

                if (use_icount) {

                    int64_t add;

                    int64_t delta;

                    /* Advance virtual time to the next event.  */

                    if (use_icount == 1) {

                        /* When not using an adaptive execution frequency

                           we tend to get badly out of sync with real time,

                           so just delay for a reasonable amount of time.  */

                        delta = 0;

                    } else {

                        delta = cpu_get_icount() - cpu_get_clock();

                    }

                    if (delta > 0) {

                        /* If virtual time is ahead of real time then just

                           wait for IO.  */

                        timeout = (delta / 1000000) + 1;

                    } else {

                        /* Wait for either IO to occur or the next

                           timer event.  */

                        add = qemu_next_deadline();

                        /* We advance the timer before checking for IO.

                           Limit the amount we advance so that early IO

                           activity won't get the guest too far ahead.  */

                        if (add > 10000000)

                            add = 10000000;

                        delta += add;

                        add = (add + (1 << icount_time_shift) - 1)

                              >> icount_time_shift;

                        qemu_icount += add;

                        timeout = delta / 1000000;

                        if (timeout < 0)

                            timeout = 0;

                    }

                } else {

                    timeout = 10;

                }

            } else {

                timeout = 0;

            }

        } else {

            if (shutdown_requested)

                break;

            timeout = 10;

        }

#ifdef CONFIG_PROFILER

        ti = profile_getclock();

#endif

        main_loop_wait(timeout);

#ifdef CONFIG_PROFILER

        dev_time += profile_getclock() - ti;

#endif

    }

    cpu_disable_ticks();

    return ret;

}
