void do_interrupt (CPUState *env)

{

#if defined (CONFIG_USER_ONLY)

    env->exception_index |= 0x100;

#else

    uint32_t msr;

    int excp = env->exception_index;



    msr = _load_msr(env);

#if defined (DEBUG_EXCEPTIONS)

    if ((excp == EXCP_PROGRAM || excp == EXCP_DSI) && msr_pr == 1) 

    {

        if (loglevel > 0) {

            fprintf(logfile, "Raise exception at 0x%08x => 0x%08x (%02x)\n",

                    env->nip, excp << 8, env->error_code);

    }

	if (loglevel > 0)

	    cpu_ppc_dump_state(env, logfile, 0);

    }

#endif

    /* Generate informations in save/restore registers */

    switch (excp) {

    case EXCP_OFCALL:

#if defined (USE_OPEN_FIRMWARE)

        env->gpr[3] = OF_client_entry((void *)env->gpr[3]);

#endif

        return;

    case EXCP_RTASCALL:

#if defined (USE_OPEN_FIRMWARE)

        printf("RTAS call !\n");

        env->gpr[3] = RTAS_entry((void *)env->gpr[3]);

        printf("RTAS call done\n");

#endif

        return;

    case EXCP_NONE:

        /* Do nothing */

#if defined (DEBUG_EXCEPTIONS)

        printf("%s: escape EXCP_NONE\n", __func__);

#endif

        return;

    case EXCP_RESET:

        if (msr_ip)

            excp += 0xFFC00;

        goto store_next;

    case EXCP_MACHINE_CHECK:

        if (msr_me == 0) {

            cpu_abort(env, "Machine check exception while not allowed\n");

        }

        msr_me = 0;

        break;

    case EXCP_DSI:

        /* Store exception cause */

        /* data location address has been stored

         * when the fault has been detected

     */

	msr &= ~0xFFFF0000;

	env->spr[DSISR] = 0;

	if (env->error_code &  EXCP_DSI_TRANSLATE)

	    env->spr[DSISR] |= 0x40000000;

	else if (env->error_code & EXCP_DSI_PROT)

	    env->spr[DSISR] |= 0x08000000;

	else if (env->error_code & EXCP_DSI_NOTSUP) {

	    env->spr[DSISR] |= 0x80000000;

	    if (env->error_code & EXCP_DSI_DIRECT)

		env->spr[DSISR] |= 0x04000000;

	}

	if (env->error_code & EXCP_DSI_STORE)

	    env->spr[DSISR] |= 0x02000000;

	if ((env->error_code & 0xF) == EXCP_DSI_DABR)

	    env->spr[DSISR] |= 0x00400000;

	if (env->error_code & EXCP_DSI_ECXW)

	    env->spr[DSISR] |= 0x00100000;

#if defined (DEBUG_EXCEPTIONS)

	if (loglevel) {

	    fprintf(logfile, "DSI exception: DSISR=0x%08x, DAR=0x%08x\n",

		    env->spr[DSISR], env->spr[DAR]);

	} else {

	    printf("DSI exception: DSISR=0x%08x, DAR=0x%08x nip=0x%08x\n",

		   env->spr[DSISR], env->spr[DAR], env->nip);

	}

#endif

        goto store_next;

    case EXCP_ISI:

        /* Store exception cause */

	msr &= ~0xFFFF0000;

        if (env->error_code == EXCP_ISI_TRANSLATE)

            msr |= 0x40000000;

        else if (env->error_code == EXCP_ISI_NOEXEC ||

		 env->error_code == EXCP_ISI_GUARD ||

		 env->error_code == EXCP_ISI_DIRECT)

            msr |= 0x10000000;

        else

            msr |= 0x08000000;

#if defined (DEBUG_EXCEPTIONS)

	if (loglevel) {

	    fprintf(logfile, "ISI exception: msr=0x%08x, nip=0x%08x\n",

		    msr, env->nip);

	} else {

	    printf("ISI exception: msr=0x%08x, nip=0x%08x tbl:0x%08x\n",

		   msr, env->nip, env->spr[V_TBL]);

	}

#endif

        goto store_next;

    case EXCP_EXTERNAL:

        if (msr_ee == 0) {

#if defined (DEBUG_EXCEPTIONS)

            if (loglevel > 0) {

                fprintf(logfile, "Skipping hardware interrupt\n");

    }

#endif

            /* Requeue it */

            do_raise_exception(EXCP_EXTERNAL);

            return;

            }

        goto store_next;

    case EXCP_ALIGN:

        /* Store exception cause */

        /* Get rS/rD and rA from faulting opcode */

        env->spr[DSISR] |=

            (ldl_code((void *)(env->nip - 4)) & 0x03FF0000) >> 16;

        /* data location address has been stored

         * when the fault has been detected

         */

        goto store_current;

    case EXCP_PROGRAM:

        msr &= ~0xFFFF0000;

        switch (env->error_code & ~0xF) {

        case EXCP_FP:

            if (msr_fe0 == 0 && msr_fe1 == 0) {

#if defined (DEBUG_EXCEPTIONS)

                printf("Ignore floating point exception\n");

#endif

                return;

        }

            msr |= 0x00100000;

            /* Set FX */

            env->fpscr[7] |= 0x8;

            /* Finally, update FEX */

            if ((((env->fpscr[7] & 0x3) << 3) | (env->fpscr[6] >> 1)) &

                ((env->fpscr[1] << 1) | (env->fpscr[0] >> 3)))

                env->fpscr[7] |= 0x4;

        break;

        case EXCP_INVAL:

            //	    printf("Invalid instruction at 0x%08x\n", env->nip);

            msr |= 0x00080000;

        break;

        case EXCP_PRIV:

            msr |= 0x00040000;

        break;

        case EXCP_TRAP:

            msr |= 0x00020000;

            break;

        default:

            /* Should never occur */

        break;

    }

        msr |= 0x00010000;

        goto store_current;

    case EXCP_NO_FP:

        goto store_current;

    case EXCP_DECR:

        if (msr_ee == 0) {

            /* Requeue it */

            do_raise_exception(EXCP_DECR);

            return;

        }

        goto store_next;

    case EXCP_SYSCALL:

#if defined (DEBUG_EXCEPTIONS)

	if (msr_pr) {

	    if (loglevel) {

		fprintf(logfile, "syscall %d 0x%08x 0x%08x 0x%08x 0x%08x\n",

			env->gpr[0], env->gpr[3], env->gpr[4],

			env->gpr[5], env->gpr[6]);

	    } else {

		printf("syscall %d from 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",

		       env->gpr[0], env->nip, env->gpr[3], env->gpr[4],

		       env->gpr[5], env->gpr[6]);

	    }

	}

#endif

        goto store_next;

    case EXCP_TRACE:

        goto store_next;

    case EXCP_FP_ASSIST:

        goto store_next;

    case EXCP_MTMSR:

        /* Nothing to do */

        return;

    case EXCP_BRANCH:

        /* Nothing to do */

        return;

    case EXCP_RFI:

        /* Restore user-mode state */

	tb_flush(env);

#if defined (DEBUG_EXCEPTIONS)

	if (msr_pr == 1)

	    printf("Return from exception => 0x%08x\n", (uint32_t)env->nip);

#endif

        return;

    store_current:

        /* SRR0 is set to current instruction */

        env->spr[SRR0] = (uint32_t)env->nip - 4;

        break;

    store_next:

        /* SRR0 is set to next instruction */

        env->spr[SRR0] = (uint32_t)env->nip;

        break;

    }

    env->spr[SRR1] = msr;

    /* reload MSR with correct bits */

    msr_pow = 0;

    msr_ee = 0;

    msr_pr = 0;

    msr_fp = 0;

    msr_fe0 = 0;

    msr_se = 0;

    msr_be = 0;

    msr_fe1 = 0;

    msr_ir = 0;

    msr_dr = 0;

    msr_ri = 0;

    msr_le = msr_ile;

    /* Jump to handler */

    env->nip = excp << 8;

    env->exception_index = EXCP_NONE;

    /* Invalidate all TLB as we may have changed translation mode */

    tlb_flush(env, 1);

    /* ensure that no TB jump will be modified as

       the program flow was changed */

#ifdef __sparc__

    tmp_T0 = 0;

#else

    T0 = 0;

#endif

#endif

    env->exception_index = -1;

}
