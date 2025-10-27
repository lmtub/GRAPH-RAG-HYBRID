static void setup_rt_frame(int usig, struct emulated_sigaction *ka,

                           target_siginfo_t *info,

			   target_sigset_t *set, CPUState *env)

{

	struct rt_sigframe *frame;

	abi_ulong frame_addr = get_sigframe(ka, env, sizeof(*frame));

	struct target_sigaltstack stack;

	int i, err = 0;



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

            return /* 1 */;



	__put_user_error(&frame->info, (abi_ulong *)&frame->pinfo, err);

	__put_user_error(&frame->uc, (abi_ulong *)&frame->puc, err);

	err |= copy_siginfo_to_user(&frame->info, info);



	/* Clear all the bits of the ucontext we don't use.  */

	memset(&frame->uc, 0, offsetof(struct target_ucontext, tuc_mcontext));



        memset(&stack, 0, sizeof(stack));

        __put_user(target_sigaltstack_used.ss_sp, &stack.ss_sp);

        __put_user(target_sigaltstack_used.ss_size, &stack.ss_size);

        __put_user(sas_ss_flags(get_sp_from_cpustate(env)), &stack.ss_flags);

        memcpy(&frame->uc.tuc_stack, &stack, sizeof(stack));



	err |= setup_sigcontext(&frame->uc.tuc_mcontext, /*&frame->fpstate,*/

				env, set->sig[0]);

        for(i = 0; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]))

                goto end;

        }



	if (err == 0)

		err = setup_return(env, ka, &frame->retcode, frame, usig);



	if (err == 0) {

		/*

		 * For realtime signals we must also set the second and third

		 * arguments for the signal handler.

		 *   -- Peter Maydell <pmaydell@chiark.greenend.org.uk> 2000-12-06

		 */

            env->regs[1] = (abi_ulong)frame->pinfo;

            env->regs[2] = (abi_ulong)frame->puc;

	}



end:

	unlock_user_struct(frame, frame_addr, 1);



        //	return err;

}
