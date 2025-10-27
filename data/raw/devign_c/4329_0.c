static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUSH4State *regs)

{

    struct target_sigframe *frame;

    abi_ulong frame_addr;

    int i;

    int err = 0;

    int signal;



    frame_addr = get_sigframe(ka, regs->gregs[15], sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

	goto give_sigsegv;



    signal = current_exec_domain_sig(sig);



    err |= setup_sigcontext(&frame->sc, regs, set->sig[0]);



    for (i = 0; i < TARGET_NSIG_WORDS - 1; i++) {

        __put_user(set->sig[i + 1], &frame->extramask[i]);

    }



    /* Set up to return from userspace.  If provided, use a stub

       already in userspace.  */

    if (ka->sa_flags & TARGET_SA_RESTORER) {

        regs->pr = (unsigned long) ka->sa_restorer;

    } else {

        /* Generate return code (system call to sigreturn) */

        __put_user(MOVW(2), &frame->retcode[0]);

        __put_user(TRAP_NOARG, &frame->retcode[1]);

        __put_user((TARGET_NR_sigreturn), &frame->retcode[2]);

        regs->pr = (unsigned long) frame->retcode;

    }



    if (err)

        goto give_sigsegv;



    /* Set up registers for signal handler */

    regs->gregs[15] = frame_addr;

    regs->gregs[4] = signal; /* Arg for signal handler */

    regs->gregs[5] = 0;

    regs->gregs[6] = frame_addr += offsetof(typeof(*frame), sc);

    regs->pc = (unsigned long) ka->_sa_handler;



    unlock_user_struct(frame, frame_addr, 1);

    return;



give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
