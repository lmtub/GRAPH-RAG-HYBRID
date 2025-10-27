static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUM68KState *env)

{

    struct target_sigframe *frame;

    abi_ulong frame_addr;

    abi_ulong retcode_addr;

    abi_ulong sc_addr;

    int err = 0;

    int i;



    frame_addr = get_sigframe(ka, env, sizeof *frame);

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

	goto give_sigsegv;



    __put_user(sig, &frame->sig);



    sc_addr = frame_addr + offsetof(struct target_sigframe, sc);

    __put_user(sc_addr, &frame->psc);



    setup_sigcontext(&frame->sc, env, set->sig[0]);



    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        if (__put_user(set->sig[i], &frame->extramask[i - 1]))

            goto give_sigsegv;

    }



    /* Set up to return from userspace.  */



    retcode_addr = frame_addr + offsetof(struct target_sigframe, retcode);

    __put_user(retcode_addr, &frame->pretcode);



    /* moveq #,d0; trap #0 */



    __put_user(0x70004e40 + (TARGET_NR_sigreturn << 16),

                      (long *)(frame->retcode));



    if (err)

        goto give_sigsegv;



    /* Set up to return from userspace */



    env->aregs[7] = frame_addr;

    env->pc = ka->_sa_handler;



    unlock_user_struct(frame, frame_addr, 1);

    return;



give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
