long do_rt_sigreturn(CPUTLGState *env)

{

    abi_ulong frame_addr = env->regs[TILEGX_R_SP];

    struct target_rt_sigframe *frame;

    sigset_t set;



    trace_user_do_rt_sigreturn(env, frame_addr);

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }

    target_to_host_sigset(&set, &frame->uc.tuc_sigmask);

    do_sigprocmask(SIG_SETMASK, &set, NULL);



    restore_sigcontext(env, &frame->uc.tuc_mcontext);

    if (do_sigaltstack(frame_addr + offsetof(struct target_rt_sigframe,

                                             uc.tuc_stack),

                       0, env->regs[TILEGX_R_SP]) == -EFAULT) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return env->regs[TILEGX_R_RE];





 badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

}
