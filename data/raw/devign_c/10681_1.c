restore_sigcontext(CPUX86State *env, struct target_sigcontext *sc, int *peax)

{

	unsigned int err = 0;

        abi_ulong fpstate_addr;

        unsigned int tmpflags;



        cpu_x86_load_seg(env, R_GS, tswap16(sc->gs));

        cpu_x86_load_seg(env, R_FS, tswap16(sc->fs));

        cpu_x86_load_seg(env, R_ES, tswap16(sc->es));

        cpu_x86_load_seg(env, R_DS, tswap16(sc->ds));



        env->regs[R_EDI] = tswapl(sc->edi);

        env->regs[R_ESI] = tswapl(sc->esi);

        env->regs[R_EBP] = tswapl(sc->ebp);

        env->regs[R_ESP] = tswapl(sc->esp);

        env->regs[R_EBX] = tswapl(sc->ebx);

        env->regs[R_EDX] = tswapl(sc->edx);

        env->regs[R_ECX] = tswapl(sc->ecx);

        env->eip = tswapl(sc->eip);



        cpu_x86_load_seg(env, R_CS, lduw(&sc->cs) | 3);

        cpu_x86_load_seg(env, R_SS, lduw(&sc->ss) | 3);



        tmpflags = tswapl(sc->eflags);

        env->eflags = (env->eflags & ~0x40DD5) | (tmpflags & 0x40DD5);

        //		regs->orig_eax = -1;		/* disable syscall checks */



        fpstate_addr = tswapl(sc->fpstate);

	if (fpstate_addr != 0) {

                if (!access_ok(VERIFY_READ, fpstate_addr, 

                               sizeof(struct target_fpstate)))

                        goto badframe;

                cpu_x86_frstor(env, fpstate_addr, 1);

	}



        *peax = tswapl(sc->eax);

	return err;

badframe:

	return 1;

}
