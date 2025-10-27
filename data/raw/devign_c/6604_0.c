long do_sigreturn(CPUState *env)

{

        struct target_signal_frame *sf;

        uint32_t up_psr, pc, npc;

        target_sigset_t set;

        sigset_t host_set;

        abi_ulong fpu_save;

        int err, i;



        sf = (struct target_signal_frame *)g2h(env->regwptr[UREG_FP]);

#if 0

	fprintf(stderr, "sigreturn\n");

	fprintf(stderr, "sf: %x pc %x fp %x sp %x\n", sf, env->pc, env->regwptr[UREG_FP], env->regwptr[UREG_SP]);

#endif

	//cpu_dump_state(env, stderr, fprintf, 0);



        /* 1. Make sure we are not getting garbage from the user */

#if 0

        if (verify_area (VERIFY_READ, sf, sizeof (*sf)))

                goto segv_and_exit;

#endif



        if (((uint) sf) & 3)

                goto segv_and_exit;



        err = __get_user(pc,  &sf->info.si_regs.pc);

        err |= __get_user(npc, &sf->info.si_regs.npc);



        if ((pc | npc) & 3)

                goto segv_and_exit;



        /* 2. Restore the state */

        err |= __get_user(up_psr, &sf->info.si_regs.psr);



        /* User can only change condition codes and FPU enabling in %psr. */

        env->psr = (up_psr & (PSR_ICC /* | PSR_EF */))

                  | (env->psr & ~(PSR_ICC /* | PSR_EF */));



	env->pc = pc;

	env->npc = npc;

        err |= __get_user(env->y, &sf->info.si_regs.y);

	for (i=0; i < 8; i++) {

		err |= __get_user(env->gregs[i], &sf->info.si_regs.u_regs[i]);

	}

	for (i=0; i < 8; i++) {

		err |= __get_user(env->regwptr[i + UREG_I0], &sf->info.si_regs.u_regs[i+8]);

	}



        err |= __get_user(fpu_save, (abi_ulong *)&sf->fpu_save);



        //if (fpu_save)

        //        err |= restore_fpu_state(env, fpu_save);



        /* This is pretty much atomic, no amount locking would prevent

         * the races which exist anyways.

         */

        err |= __get_user(set.sig[0], &sf->info.si_mask);

        for(i = 1; i < TARGET_NSIG_WORDS; i++) {

            err |= (__get_user(set.sig[i], &sf->extramask[i - 1]));

        }



        target_to_host_sigset_internal(&host_set, &set);

        sigprocmask(SIG_SETMASK, &host_set, NULL);



        if (err)

                goto segv_and_exit;



        return env->regwptr[0];



segv_and_exit:

	force_sig(TARGET_SIGSEGV);

}
