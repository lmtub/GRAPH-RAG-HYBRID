static void handle_pending_signal(CPUArchState *cpu_env, int sig,

                                  struct emulated_sigtable *k)

{

    CPUState *cpu = ENV_GET_CPU(cpu_env);

    abi_ulong handler;

    sigset_t set;

    target_sigset_t target_old_set;

    struct target_sigaction *sa;

    TaskState *ts = cpu->opaque;



    trace_user_handle_signal(cpu_env, sig);

    /* dequeue signal */

    k->pending = 0;



    sig = gdb_handlesig(cpu, sig);

    if (!sig) {

        sa = NULL;

        handler = TARGET_SIG_IGN;

    } else {

        sa = &sigact_table[sig - 1];

        handler = sa->_sa_handler;

    }



    if (do_strace) {

        print_taken_signal(sig, &k->info);

    }



    if (handler == TARGET_SIG_DFL) {

        /* default handler : ignore some signal. The other are job control or fatal */

        if (sig == TARGET_SIGTSTP || sig == TARGET_SIGTTIN || sig == TARGET_SIGTTOU) {

            kill(getpid(),SIGSTOP);

        } else if (sig != TARGET_SIGCHLD &&

                   sig != TARGET_SIGURG &&

                   sig != TARGET_SIGWINCH &&

                   sig != TARGET_SIGCONT) {

            force_sig(sig);

        }

    } else if (handler == TARGET_SIG_IGN) {

        /* ignore sig */

    } else if (handler == TARGET_SIG_ERR) {

        force_sig(sig);

    } else {

        /* compute the blocked signals during the handler execution */

        sigset_t *blocked_set;



        target_to_host_sigset(&set, &sa->sa_mask);

        /* SA_NODEFER indicates that the current signal should not be

           blocked during the handler */

        if (!(sa->sa_flags & TARGET_SA_NODEFER))

            sigaddset(&set, target_to_host_signal(sig));



        /* save the previous blocked signal state to restore it at the

           end of the signal execution (see do_sigreturn) */

        host_to_target_sigset_internal(&target_old_set, &ts->signal_mask);



        /* block signals in the handler */

        blocked_set = ts->in_sigsuspend ?

            &ts->sigsuspend_mask : &ts->signal_mask;

        sigorset(&ts->signal_mask, blocked_set, &set);

        ts->in_sigsuspend = 0;



        /* if the CPU is in VM86 mode, we restore the 32 bit values */

#if defined(TARGET_I386) && !defined(TARGET_X86_64)

        {

            CPUX86State *env = cpu_env;

            if (env->eflags & VM_MASK)

                save_v86_state(env);

        }

#endif

        /* prepare the stack frame of the virtual CPU */

#if defined(TARGET_ABI_MIPSN32) || defined(TARGET_ABI_MIPSN64) \

    || defined(TARGET_OPENRISC) || defined(TARGET_TILEGX)

        /* These targets do not have traditional signals.  */

        setup_rt_frame(sig, sa, &k->info, &target_old_set, cpu_env);

#else

        if (sa->sa_flags & TARGET_SA_SIGINFO)

            setup_rt_frame(sig, sa, &k->info, &target_old_set, cpu_env);

        else

            setup_frame(sig, sa, &target_old_set, cpu_env);

#endif

        if (sa->sa_flags & TARGET_SA_RESETHAND) {

            sa->_sa_handler = TARGET_SIG_DFL;

        }

    }

}
