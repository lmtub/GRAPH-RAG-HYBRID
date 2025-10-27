setup_sigcontext(struct target_sigcontext *sc, struct target_fpstate *fpstate,

		 CPUX86State *env, abi_ulong mask, abi_ulong fpstate_addr)

{

    CPUState *cs = CPU(x86_env_get_cpu(env));

    int err = 0;

    uint16_t magic;



	/* already locked in setup_frame() */

    __put_user(env->segs[R_GS].selector, (unsigned int *)&sc->gs);

    __put_user(env->segs[R_FS].selector, (unsigned int *)&sc->fs);

    __put_user(env->segs[R_ES].selector, (unsigned int *)&sc->es);

    __put_user(env->segs[R_DS].selector, (unsigned int *)&sc->ds);

    __put_user(env->regs[R_EDI], &sc->edi);

    __put_user(env->regs[R_ESI], &sc->esi);

    __put_user(env->regs[R_EBP], &sc->ebp);

    __put_user(env->regs[R_ESP], &sc->esp);

    __put_user(env->regs[R_EBX], &sc->ebx);

    __put_user(env->regs[R_EDX], &sc->edx);

    __put_user(env->regs[R_ECX], &sc->ecx);

    __put_user(env->regs[R_EAX], &sc->eax);

    __put_user(cs->exception_index, &sc->trapno);

    __put_user(env->error_code, &sc->err);

    __put_user(env->eip, &sc->eip);

    __put_user(env->segs[R_CS].selector, (unsigned int *)&sc->cs);

    __put_user(env->eflags, &sc->eflags);

    __put_user(env->regs[R_ESP], &sc->esp_at_signal);

    __put_user(env->segs[R_SS].selector, (unsigned int *)&sc->ss);



        cpu_x86_fsave(env, fpstate_addr, 1);

        fpstate->status = fpstate->sw;

        magic = 0xffff;

    __put_user(magic, &fpstate->magic);

    __put_user(fpstate_addr, &sc->fpstate);



	/* non-iBCS2 extensions.. */

    __put_user(mask, &sc->oldmask);

    __put_user(env->cr[2], &sc->cr2);

	return err;

}
