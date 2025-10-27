long do_sigreturn(CPUCRISState *env)

{

	struct target_signal_frame *frame;

	abi_ulong frame_addr;

	target_sigset_t target_set;

	sigset_t set;

	int i;



	frame_addr = env->regs[R_SP];

	/* Make sure the guest isn't playing games.  */

	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 1))

		goto badframe;



	/* Restore blocked signals */

	if (__get_user(target_set.sig[0], &frame->sc.oldmask))

		goto badframe;

	for(i = 1; i < TARGET_NSIG_WORDS; i++) {

		if (__get_user(target_set.sig[i], &frame->extramask[i - 1]))

			goto badframe;

	}

	target_to_host_sigset_internal(&set, &target_set);

        do_sigprocmask(SIG_SETMASK, &set, NULL);



	restore_sigcontext(&frame->sc, env);

	unlock_user_struct(frame, frame_addr, 0);

	return env->regs[10];

  badframe:

	unlock_user_struct(frame, frame_addr, 0);

	force_sig(TARGET_SIGSEGV);

}
