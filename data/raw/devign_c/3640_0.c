static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

                           target_sigset_t *set, CPUSH4State *regs)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr;

    int i;

    int err = 0;

    int signal;



    frame_addr = get_sigframe(ka, regs->gregs[15], sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

	goto give_sigsegv;



    signal = current_exec_domain_sig(sig);



    err |= copy_siginfo_to_user(&frame->info, info);



    /* Create the ucontext.  */

    __put_user(0, &frame->uc.tuc_flags);

    __put_user(0, (unsigned long *)&frame->uc.tuc_link);

    __put_user((unsigned long)target_sigaltstack_used.ss_sp,

               &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(regs->gregs[15]),

               &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

               &frame->uc.tuc_stack.ss_size);

    setup_sigcontext(&frame->uc.tuc_mcontext,

			    regs, set->sig[0]);

    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

        __put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]);

    }



    /* Set up to return from userspace.  If provided, use a stub

       already in userspace.  */

    if (ka->sa_flags & TARGET_SA_RESTORER) {

        regs->pr = (unsigned long) ka->sa_restorer;

    } else {

        /* Generate return code (system call to sigreturn) */

        __put_user(MOVW(2), &frame->retcode[0]);

        __put_user(TRAP_NOARG, &frame->retcode[1]);

        __put_user((TARGET_NR_rt_sigreturn), &frame->retcode[2]);

        regs->pr = (unsigned long) frame->retcode;

    }



    if (err)

        goto give_sigsegv;



    /* Set up registers for signal handler */

    regs->gregs[15] = frame_addr;

    regs->gregs[4] = signal; /* Arg for signal handler */

    regs->gregs[5] = frame_addr + offsetof(typeof(*frame), info);

    regs->gregs[6] = frame_addr + offsetof(typeof(*frame), uc);

    regs->pc = (unsigned long) ka->_sa_handler;



    unlock_user_struct(frame, frame_addr, 1);

    return;



give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
