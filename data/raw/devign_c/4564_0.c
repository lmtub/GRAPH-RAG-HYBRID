static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

                           target_sigset_t *set, CPUM68KState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr;

    abi_ulong retcode_addr;

    abi_ulong info_addr;

    abi_ulong uc_addr;

    int err = 0;

    int i;



    frame_addr = get_sigframe(ka, env, sizeof *frame);

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

	goto give_sigsegv;



    __put_user(sig, &frame->sig);



    info_addr = frame_addr + offsetof(struct target_rt_sigframe, info);

    __put_user(info_addr, &frame->pinfo);



    uc_addr = frame_addr + offsetof(struct target_rt_sigframe, uc);

    __put_user(uc_addr, &frame->puc);



    err |= copy_siginfo_to_user(&frame->info, info);



    /* Create the ucontext */



    __put_user(0, &frame->uc.tuc_flags);

    __put_user(0, &frame->uc.tuc_link);

    __put_user(target_sigaltstack_used.ss_sp,

               &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(env->aregs[7]),

               &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

               &frame->uc.tuc_stack.ss_size);

    err |= target_rt_setup_ucontext(&frame->uc, env);



    if (err)

            goto give_sigsegv;



    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

        if (__put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]))

            goto give_sigsegv;

    }



    /* Set up to return from userspace.  */



    retcode_addr = frame_addr + offsetof(struct target_sigframe, retcode);

    __put_user(retcode_addr, &frame->pretcode);



    /* moveq #,d0; notb d0; trap #0 */



    __put_user(0x70004600 + ((TARGET_NR_rt_sigreturn ^ 0xff) << 16),

               (long *)(frame->retcode + 0));

    __put_user(0x4e40, (short *)(frame->retcode + 4));



    if (err)

        goto give_sigsegv;



    /* Set up to return from userspace */



    env->aregs[7] = frame_addr;

    env->pc = ka->_sa_handler;



    unlock_user_struct(frame, frame_addr, 1);

    return;



give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
