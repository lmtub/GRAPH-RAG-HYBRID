static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUCRISState *env)

{

	struct target_signal_frame *frame;

	abi_ulong frame_addr;

	int err = 0;

	int i;



	frame_addr = get_sigframe(env, sizeof *frame);

	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

		goto badframe;



	/*

	 * The CRIS signal return trampoline. A real linux/CRIS kernel doesn't

	 * use this trampoline anymore but it sets it up for GDB.

	 * In QEMU, using the trampoline simplifies things a bit so we use it.

	 *

	 * This is movu.w __NR_sigreturn, r9; break 13;

	 */

    __put_user(0x9c5f, frame->retcode+0);

    __put_user(TARGET_NR_sigreturn,

               frame->retcode + 1);

    __put_user(0xe93d, frame->retcode + 2);



	/* Save the mask.  */

    __put_user(set->sig[0], &frame->sc.oldmask);

	if (err)

		goto badframe;



	for(i = 1; i < TARGET_NSIG_WORDS; i++) {

		if (__put_user(set->sig[i], &frame->extramask[i - 1]))

			goto badframe;

	}



	setup_sigcontext(&frame->sc, env);



	/* Move the stack and setup the arguments for the handler.  */

	env->regs[R_SP] = frame_addr;

	env->regs[10] = sig;

	env->pc = (unsigned long) ka->_sa_handler;

	/* Link SRP so the guest returns through the trampoline.  */

	env->pregs[PR_SRP] = frame_addr + offsetof(typeof(*frame), retcode);



	unlock_user_struct(frame, frame_addr, 1);

	return;

  badframe:

	unlock_user_struct(frame, frame_addr, 1);

	force_sig(TARGET_SIGSEGV);

}
