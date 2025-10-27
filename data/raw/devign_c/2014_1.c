long do_sigreturn(CPUS390XState *env)

{

    sigframe *frame;

    abi_ulong frame_addr = env->regs[15];

    target_sigset_t target_set;

    sigset_t set;



    trace_user_do_sigreturn(env, frame_addr);

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }

    __get_user(target_set.sig[0], &frame->sc.oldmask[0]);



    target_to_host_sigset_internal(&set, &target_set);

    set_sigmask(&set); /* ~_BLOCKABLE? */



    if (restore_sigregs(env, &frame->sregs)) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    force_sig(TARGET_SIGSEGV);

    return 0;

}
