long do_rt_sigreturn(CPUX86State *env)

{

    abi_ulong frame_addr;

    struct rt_sigframe *frame;

    sigset_t set;



    frame_addr = env->regs[R_ESP] - 4;

    trace_user_do_rt_sigreturn(env, frame_addr);

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

        goto badframe;

    target_to_host_sigset(&set, &frame->uc.tuc_sigmask);

    set_sigmask(&set);



    if (restore_sigcontext(env, &frame->uc.tuc_mcontext)) {

        goto badframe;

    }



    if (do_sigaltstack(frame_addr + offsetof(struct rt_sigframe, uc.tuc_stack), 0,

                       get_sp_from_cpustate(env)) == -EFAULT) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
