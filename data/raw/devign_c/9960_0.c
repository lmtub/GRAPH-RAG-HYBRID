long do_sigreturn(CPUSH4State *regs)

{

    struct target_sigframe *frame;

    abi_ulong frame_addr;

    sigset_t blocked;

    target_sigset_t target_set;

    target_ulong r0;

    int i;

    int err = 0;



#if defined(DEBUG_SIGNAL)

    fprintf(stderr, "do_sigreturn\n");

#endif

    frame_addr = regs->gregs[15];

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

   	goto badframe;



    err |= __get_user(target_set.sig[0], &frame->sc.oldmask);

    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        err |= (__get_user(target_set.sig[i], &frame->extramask[i - 1]));

    }



    if (err)

        goto badframe;



    target_to_host_sigset_internal(&blocked, &target_set);

    sigprocmask(SIG_SETMASK, &blocked, NULL);



    if (restore_sigcontext(regs, &frame->sc, &r0))

        goto badframe;



    unlock_user_struct(frame, frame_addr, 0);

    return r0;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
