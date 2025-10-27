static int do_sigframe_return_v2(CPUARMState *env, target_ulong frame_addr,

                                 struct target_ucontext_v2 *uc)

{

    sigset_t host_set;

    abi_ulong *regspace;



    target_to_host_sigset(&host_set, &uc->tuc_sigmask);

    sigprocmask(SIG_SETMASK, &host_set, NULL);



    if (restore_sigcontext(env, &uc->tuc_mcontext))

        return 1;



    /* Restore coprocessor signal frame */

    regspace = uc->tuc_regspace;

    if (arm_feature(env, ARM_FEATURE_VFP)) {

        regspace = restore_sigframe_v2_vfp(env, regspace);

        if (!regspace) {

            return 1;

        }

    }

    if (arm_feature(env, ARM_FEATURE_IWMMXT)) {

        regspace = restore_sigframe_v2_iwmmxt(env, regspace);

        if (!regspace) {

            return 1;

        }

    }



    if (do_sigaltstack(frame_addr + offsetof(struct target_ucontext_v2, tuc_stack), 0, get_sp_from_cpustate(env)) == -EFAULT)

        return 1;



#if 0

    /* Send SIGTRAP if we're single-stepping */

    if (ptrace_cancel_bpt(current))

            send_sig(SIGTRAP, current, 1);

#endif



    return 0;

}
