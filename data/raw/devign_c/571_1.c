static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

                           target_sigset_t *set, CPUS390XState *env)

{

    int i;

    rt_sigframe *frame;

    abi_ulong frame_addr;



    frame_addr = get_sigframe(ka, env, sizeof *frame);

    qemu_log("%s: frame_addr 0x%llx\n", __FUNCTION__,

             (unsigned long long)frame_addr);

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0)) {

        goto give_sigsegv;

    }



    qemu_log("%s: 1\n", __FUNCTION__);

    copy_siginfo_to_user(&frame->info, info);



    /* Create the ucontext.  */

    __put_user(0, &frame->uc.tuc_flags);

    __put_user((abi_ulong)0, (abi_ulong *)&frame->uc.tuc_link);

    __put_user(target_sigaltstack_used.ss_sp, &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(get_sp_from_cpustate(env)),

                      &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size, &frame->uc.tuc_stack.ss_size);

    save_sigregs(env, &frame->uc.tuc_mcontext);

    for (i = 0; i < TARGET_NSIG_WORDS; i++) {

        __put_user((abi_ulong)set->sig[i],

        (abi_ulong *)&frame->uc.tuc_sigmask.sig[i]);

    }



    /* Set up to return from userspace.  If provided, use a stub

       already in userspace.  */

    if (ka->sa_flags & TARGET_SA_RESTORER) {

        env->regs[14] = (unsigned long) ka->sa_restorer | PSW_ADDR_AMODE;

    } else {

        env->regs[14] = (unsigned long) frame->retcode | PSW_ADDR_AMODE;

        if (__put_user(S390_SYSCALL_OPCODE | TARGET_NR_rt_sigreturn,

                       (uint16_t *)(frame->retcode))) {

            goto give_sigsegv;

        }

    }



    /* Set up backchain. */

    if (__put_user(env->regs[15], (abi_ulong *) frame)) {

        goto give_sigsegv;

    }



    /* Set up registers for signal handler */

    env->regs[15] = frame_addr;

    env->psw.addr = (target_ulong) ka->_sa_handler | PSW_ADDR_AMODE;



    env->regs[2] = sig; //map_signal(sig);

    env->regs[3] = frame_addr + offsetof(typeof(*frame), info);

    env->regs[4] = frame_addr + offsetof(typeof(*frame), uc);

    return;



give_sigsegv:

    qemu_log("%s: give_sigsegv\n", __FUNCTION__);

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
