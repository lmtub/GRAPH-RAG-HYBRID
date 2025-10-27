setup_return(CPUARMState *env, struct target_sigaction *ka,

	     abi_ulong *rc, abi_ulong frame_addr, int usig, abi_ulong rc_addr)

{

	abi_ulong handler = ka->_sa_handler;

	abi_ulong retcode;

	int thumb = handler & 1;

	uint32_t cpsr = cpsr_read(env);



	cpsr &= ~CPSR_IT;

	if (thumb) {

		cpsr |= CPSR_T;

	} else {

		cpsr &= ~CPSR_T;

	}



	if (ka->sa_flags & TARGET_SA_RESTORER) {

		retcode = ka->sa_restorer;

	} else {

		unsigned int idx = thumb;



		if (ka->sa_flags & TARGET_SA_SIGINFO)

			idx += 2;



		if (__put_user(retcodes[idx], rc))

			return 1;



		retcode = rc_addr + thumb;

	}



	env->regs[0] = usig;

	env->regs[13] = frame_addr;

	env->regs[14] = retcode;

	env->regs[15] = handler & (thumb ? ~1 : ~3);

	cpsr_write(env, cpsr, 0xffffffff);



	return 0;

}
