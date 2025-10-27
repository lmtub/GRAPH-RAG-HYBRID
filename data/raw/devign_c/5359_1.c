long do_rt_sigreturn(CPUAlphaState *env)

{

    abi_ulong frame_addr = env->ir[IR_A0];

    struct target_rt_sigframe *frame;

    sigset_t set;



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }

    target_to_host_sigset(&set, &frame->uc.tuc_sigmask);

    do_sigprocmask(SIG_SETMASK, &set, NULL);



    if (restore_sigcontext(env, &frame->uc.tuc_mcontext)) {

        goto badframe;

    }

    if (do_sigaltstack(frame_addr + offsetof(struct target_rt_sigframe,

                                             uc.tuc_stack),

                       0, env->ir[IR_SP]) == -EFAULT) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return env->ir[IR_V0];





 badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

}
