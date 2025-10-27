long do_rt_sigreturn(CPUARMState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr = env->xregs[31];



    if (frame_addr & 15) {

        goto badframe;

    }



    if  (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }



    if (target_restore_sigframe(env, frame)) {

        goto badframe;

    }



    if (do_sigaltstack(frame_addr +

            offsetof(struct target_rt_sigframe, uc.tuc_stack),

            0, get_sp_from_cpustate(env)) == -EFAULT) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return env->xregs[0];



 badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
