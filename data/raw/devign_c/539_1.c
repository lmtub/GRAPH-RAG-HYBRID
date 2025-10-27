int cpu_exec(CPUState *cpu)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

    int ret;

    SyncClocks sc;



    /* replay_interrupt may need current_cpu */

    current_cpu = cpu;



    if (cpu_handle_halt(cpu)) {

        return EXCP_HALTED;

    }



    rcu_read_lock();



    cc->cpu_exec_enter(cpu);



    /* Calculate difference between guest clock and host clock.

     * This delay includes the delay of the last cycle, so

     * what we have to do is sleep until it is 0. As for the

     * advance/delay we gain here, we try to fix it next time.

     */

    init_delay_params(&sc, cpu);



    /* prepare setjmp context for exception handling */

    if (sigsetjmp(cpu->jmp_env, 0) != 0) {

#if defined(__clang__) || !QEMU_GNUC_PREREQ(4, 6)

        /* Some compilers wrongly smash all local variables after

         * siglongjmp. There were bug reports for gcc 4.5.0 and clang.

         * Reload essential local variables here for those compilers.

         * Newer versions of gcc would complain about this code (-Wclobbered). */

        cpu = current_cpu;

        cc = CPU_GET_CLASS(cpu);

#else /* buggy compiler */

        /* Assert that the compiler does not smash local variables. */

        g_assert(cpu == current_cpu);

        g_assert(cc == CPU_GET_CLASS(cpu));

#endif /* buggy compiler */

        cpu->can_do_io = 1;

        tb_lock_reset();

        if (qemu_mutex_iothread_locked()) {

            qemu_mutex_unlock_iothread();

        }

    }



    /* if an exception is pending, we execute it here */

    while (!cpu_handle_exception(cpu, &ret)) {

        TranslationBlock *last_tb = NULL;

        int tb_exit = 0;



        while (!cpu_handle_interrupt(cpu, &last_tb)) {

            TranslationBlock *tb = tb_find(cpu, last_tb, tb_exit);

            cpu_loop_exec_tb(cpu, tb, &last_tb, &tb_exit, &sc);

            /* Try to align the host and virtual clocks

               if the guest is in advance */

            align_clocks(&sc, cpu);

        }

    }



    cc->cpu_exec_exit(cpu);

    rcu_read_unlock();



    /* fail safe : never use current_cpu outside cpu_exec() */

    current_cpu = NULL;



    return ret;

}
