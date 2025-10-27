long do_sigreturn(CPUS390XState *env)

{

    sigframe *frame;

    abi_ulong frame_addr = env->regs[15];

    qemu_log("%s: frame_addr 0x%llx\n", __FUNCTION__,

             (unsigned long long)frame_addr);

    target_sigset_t target_set;

    sigset_t set;



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }

    if (__get_user(target_set.sig[0], &frame->sc.oldmask[0])) {

        goto badframe;

    }



    target_to_host_sigset_internal(&set, &target_set);

    sigprocmask(SIG_SETMASK, &set, NULL); /* ~_BLOCKABLE? */



    if (restore_sigregs(env, &frame->sregs)) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return env->regs[2];



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
