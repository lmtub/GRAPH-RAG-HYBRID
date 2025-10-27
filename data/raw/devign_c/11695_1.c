static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

			   target_sigset_t *set, CPUX86State *env)

{

        abi_ulong frame_addr, addr;

	struct rt_sigframe *frame;

	int i, err = 0;



	frame_addr = get_sigframe(ka, env, sizeof(*frame));



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

		goto give_sigsegv;



    __put_user(current_exec_domain_sig(sig), &frame->sig);

        addr = frame_addr + offsetof(struct rt_sigframe, info);

    __put_user(addr, &frame->pinfo);

        addr = frame_addr + offsetof(struct rt_sigframe, uc);

    __put_user(addr, &frame->puc);

	copy_siginfo_to_user(&frame->info, info);



	/* Create the ucontext.  */

    __put_user(0, &frame->uc.tuc_flags);

    __put_user(0, &frame->uc.tuc_link);

    __put_user(target_sigaltstack_used.ss_sp, &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(get_sp_from_cpustate(env)),

               &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

               &frame->uc.tuc_stack.ss_size);

    setup_sigcontext(&frame->uc.tuc_mcontext, &frame->fpstate, env,

            set->sig[0], frame_addr + offsetof(struct rt_sigframe, fpstate));



        for(i = 0; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]))

                goto give_sigsegv;

        }



	/* Set up to return from userspace.  If provided, use a stub

	   already in userspace.  */

	if (ka->sa_flags & TARGET_SA_RESTORER) {

        __put_user(ka->sa_restorer, &frame->pretcode);

	} else {

                uint16_t val16;

                addr = frame_addr + offsetof(struct rt_sigframe, retcode);

        __put_user(addr, &frame->pretcode);

		/* This is movl $,%eax ; int $0x80 */

        __put_user(0xb8, (char *)(frame->retcode+0));

        __put_user(TARGET_NR_rt_sigreturn, (int *)(frame->retcode+1));

                val16 = 0x80cd;

        __put_user(val16, (uint16_t *)(frame->retcode+5));

	}



	if (err)

		goto give_sigsegv;



	/* Set up registers for signal handler */

	env->regs[R_ESP] = frame_addr;

	env->eip = ka->_sa_handler;



        cpu_x86_load_seg(env, R_DS, __USER_DS);

        cpu_x86_load_seg(env, R_ES, __USER_DS);

        cpu_x86_load_seg(env, R_SS, __USER_DS);

        cpu_x86_load_seg(env, R_CS, __USER_CS);

	env->eflags &= ~TF_MASK;



	unlock_user_struct(frame, frame_addr, 1);



	return;



give_sigsegv:

	unlock_user_struct(frame, frame_addr, 1);

	if (sig == TARGET_SIGSEGV)

		ka->_sa_handler = TARGET_SIG_DFL;

	force_sig(TARGET_SIGSEGV /* , current */);

}
