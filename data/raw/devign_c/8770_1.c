long do_sigreturn(CPUX86State *env)

{

    struct sigframe *frame;

    abi_ulong frame_addr = env->regs[R_ESP] - 8;

    target_sigset_t target_set;

    sigset_t set;

    int eax, i;



#if defined(DEBUG_SIGNAL)

    fprintf(stderr, "do_sigreturn\n");

#endif

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

        goto badframe;

    /* set blocked signals */

    if (__get_user(target_set.sig[0], &frame->sc.oldmask))

        goto badframe;

    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        if (__get_user(target_set.sig[i], &frame->extramask[i - 1]))

            goto badframe;

    }



    target_to_host_sigset_internal(&set, &target_set);

    do_sigprocmask(SIG_SETMASK, &set, NULL);



    /* restore registers */

    if (restore_sigcontext(env, &frame->sc, &eax))

        goto badframe;

    unlock_user_struct(frame, frame_addr, 0);

    return eax;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
