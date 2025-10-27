long do_rt_sigreturn(CPUMIPSState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr;

    sigset_t blocked;



#if defined(DEBUG_SIGNAL)

    fprintf(stderr, "do_rt_sigreturn\n");

#endif

    frame_addr = env->active_tc.gpr[29];

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

   	goto badframe;



    target_to_host_sigset(&blocked, &frame->rs_uc.tuc_sigmask);

    sigprocmask(SIG_SETMASK, &blocked, NULL);



    if (restore_sigcontext(env, &frame->rs_uc.tuc_mcontext))

        goto badframe;



    if (do_sigaltstack(frame_addr +

		       offsetof(struct target_rt_sigframe, rs_uc.tuc_stack),

		       0, get_sp_from_cpustate(env)) == -EFAULT)

        goto badframe;



    env->active_tc.PC = env->CP0_EPC;

    mips_set_hflags_isa_mode_from_pc(env);

    /* I am not sure this is right, but it seems to work

    * maybe a problem with nested signals ? */

    env->CP0_EPC = 0;

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    force_sig(TARGET_SIGSEGV/*, current*/);

    return 0;

}
