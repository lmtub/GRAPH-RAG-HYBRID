long do_rt_sigreturn(CPUSH4State *regs)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr;

    sigset_t blocked;

    target_ulong r0;



    frame_addr = regs->gregs[15];

    trace_user_do_rt_sigreturn(regs, frame_addr);

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }



    target_to_host_sigset(&blocked, &frame->uc.tuc_sigmask);

    do_sigprocmask(SIG_SETMASK, &blocked, NULL);



    restore_sigcontext(regs, &frame->uc.tuc_mcontext, &r0);



    if (do_sigaltstack(frame_addr +

                       offsetof(struct target_rt_sigframe, uc.tuc_stack),

                       0, get_sp_from_cpustate(regs)) == -EFAULT) {

        goto badframe;

    }



    unlock_user_struct(frame, frame_addr, 0);

    return r0;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
