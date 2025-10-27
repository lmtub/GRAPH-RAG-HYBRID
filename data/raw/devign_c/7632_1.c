void cpu_reset (CPUCRISState *env)
{
	memset(env, 0, offsetof(CPUCRISState, breakpoints));
	tlb_flush(env, 1);
	env->pregs[PR_VR] = 32;
#if defined(CONFIG_USER_ONLY)
	/* start in user mode with interrupts enabled.  */
	env->pregs[PR_CCS] |= U_FLAG | I_FLAG;
#else
	env->pregs[PR_CCS] = 0;
#endif