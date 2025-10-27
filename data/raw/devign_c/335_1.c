static void setup_frame(int sig, struct target_sigaction *ka,

			target_sigset_t *set, CPUX86State *env)

{

	abi_ulong frame_addr;

	struct sigframe *frame;

	int i, err = 0;



	frame_addr = get_sigframe(ka, env, sizeof(*frame));



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

		goto give_sigsegv;



    __put_user(current_exec_domain_sig(sig),

               &frame->sig);

	if (err)

		goto give_sigsegv;



	setup_sigcontext(&frame->sc, &frame->fpstate, env, set->sig[0],

                         frame_addr + offsetof(struct sigframe, fpstate));

	if (err)

		goto give_sigsegv;



        for(i = 1; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->extramask[i - 1]))

                goto give_sigsegv;

        }



	/* Set up to return from userspace.  If provided, use a stub

	   already in userspace.  */

	if (ka->sa_flags & TARGET_SA_RESTORER) {

        __put_user(ka->sa_restorer, &frame->pretcode);

	} else {

                uint16_t val16;

                abi_ulong retcode_addr;

                retcode_addr = frame_addr + offsetof(struct sigframe, retcode);

        __put_user(retcode_addr, &frame->pretcode);

		/* This is popl %eax ; movl $,%eax ; int $0x80 */

                val16 = 0xb858;

        __put_user(val16, (uint16_t *)(frame->retcode+0));

        __put_user(TARGET_NR_sigreturn, (int *)(frame->retcode+2));

                val16 = 0x80cd;

        __put_user(val16, (uint16_t *)(frame->retcode+6));

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
