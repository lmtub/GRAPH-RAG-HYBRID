static void setup_rt_frame(int usig, struct emulated_sigaction *ka, 

                           target_siginfo_t *info,

			   target_sigset_t *set, CPUState *env)

{

	struct rt_sigframe *frame = get_sigframe(ka, env, sizeof(*frame));

	int err = 0;



#if 0

	if (!access_ok(VERIFY_WRITE, frame, sizeof (*frame)))

            return 1;

#endif

	__put_user_error(&frame->info, (target_ulong *)&frame->pinfo, err);

	__put_user_error(&frame->uc, (target_ulong *)&frame->puc, err);

	err |= copy_siginfo_to_user(&frame->info, info);



	/* Clear all the bits of the ucontext we don't use.  */

	err |= __clear_user(&frame->uc, offsetof(struct ucontext, uc_mcontext));



	err |= setup_sigcontext(&frame->uc.uc_mcontext, /*&frame->fpstate,*/

				env, set->sig[0]);

	err |= __copy_to_user(&frame->uc.uc_sigmask, set, sizeof(*set));



	if (err == 0)

		err = setup_return(env, ka, &frame->retcode, frame, usig);



	if (err == 0) {

		/*

		 * For realtime signals we must also set the second and third

		 * arguments for the signal handler.

		 *   -- Peter Maydell <pmaydell@chiark.greenend.org.uk> 2000-12-06

		 */

            env->regs[1] = (target_ulong)frame->pinfo;

            env->regs[2] = (target_ulong)frame->puc;

	}



        //	return err;

}
