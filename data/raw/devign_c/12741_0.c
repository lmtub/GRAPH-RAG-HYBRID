static void setup_rt_frame(int sig, struct emulated_sigaction *ka, 

                           target_siginfo_t *info,

			   target_sigset_t *set, CPUX86State *env)

{

	struct rt_sigframe *frame;

	int err = 0;



	frame = get_sigframe(ka, env, sizeof(*frame));



#if 0

	if (!access_ok(VERIFY_WRITE, frame, sizeof(*frame)))

		goto give_sigsegv;

#endif



	err |= __put_user((/*current->exec_domain

		    	   && current->exec_domain->signal_invmap

		    	   && sig < 32

		    	   ? current->exec_domain->signal_invmap[sig]

			   : */sig),

			  &frame->sig);

	err |= __put_user((target_ulong)&frame->info, &frame->pinfo);

	err |= __put_user((target_ulong)&frame->uc, &frame->puc);

	err |= copy_siginfo_to_user(&frame->info, info);

	if (err)

		goto give_sigsegv;



	/* Create the ucontext.  */

	err |= __put_user(0, &frame->uc.uc_flags);

	err |= __put_user(0, &frame->uc.uc_link);

	err |= __put_user(/*current->sas_ss_sp*/ 0, &frame->uc.uc_stack.ss_sp);

	err |= __put_user(/* sas_ss_flags(regs->esp) */ 0,

			  &frame->uc.uc_stack.ss_flags);

	err |= __put_user(/* current->sas_ss_size */ 0, &frame->uc.uc_stack.ss_size);

	err |= setup_sigcontext(&frame->uc.uc_mcontext, &frame->fpstate,

			        env, set->sig[0]);

	err |= __copy_to_user(&frame->uc.uc_sigmask, set, sizeof(*set));

	if (err)

		goto give_sigsegv;



	/* Set up to return from userspace.  If provided, use a stub

	   already in userspace.  */

	if (ka->sa.sa_flags & TARGET_SA_RESTORER) {

		err |= __put_user(ka->sa.sa_restorer, &frame->pretcode);

	} else {

		err |= __put_user(frame->retcode, &frame->pretcode);

		/* This is movl $,%eax ; int $0x80 */

		err |= __put_user(0xb8, (char *)(frame->retcode+0));

		err |= __put_user(TARGET_NR_rt_sigreturn, (int *)(frame->retcode+1));

		err |= __put_user(0x80cd, (short *)(frame->retcode+5));

	}



	if (err)

		goto give_sigsegv;



	/* Set up registers for signal handler */

	env->regs[R_ESP] = (unsigned long) frame;

	env->eip = (unsigned long) ka->sa._sa_handler;



        cpu_x86_load_seg(env, R_DS, __USER_DS);

        cpu_x86_load_seg(env, R_ES, __USER_DS);

        cpu_x86_load_seg(env, R_SS, __USER_DS);

        cpu_x86_load_seg(env, R_CS, __USER_CS);

	env->eflags &= ~TF_MASK;



	return;



give_sigsegv:

	if (sig == TARGET_SIGSEGV)

		ka->sa._sa_handler = TARGET_SIG_DFL;

	force_sig(TARGET_SIGSEGV /* , current */);

}
