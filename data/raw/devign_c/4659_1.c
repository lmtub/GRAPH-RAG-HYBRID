static void force_sigsegv(int oldsig)

{

    CPUState *cpu = thread_cpu;

    CPUArchState *env = cpu->env_ptr;

    target_siginfo_t info;



    if (oldsig == SIGSEGV) {

        /* Make sure we don't try to deliver the signal again; this will

         * end up with handle_pending_signal() calling force_sig().

         */

        sigact_table[oldsig - 1]._sa_handler = TARGET_SIG_DFL;

    }

    info.si_signo = TARGET_SIGSEGV;

    info.si_errno = 0;

    info.si_code = TARGET_SI_KERNEL;

    info._sifields._kill._pid = 0;

    info._sifields._kill._uid = 0;

    queue_signal(env, info.si_signo, QEMU_SI_KILL, &info);

}
