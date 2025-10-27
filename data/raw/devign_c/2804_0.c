void do_interrupt(CPUState *env)

{

	int ex_vec = -1;



	D(fprintf (stderr, "exception index=%d interrupt_req=%d\n",

		   env->exception_index,

		   env->interrupt_request));



	switch (env->exception_index)

	{

		case EXCP_BREAK:

			/* These exceptions are genereated by the core itself.

			   ERP should point to the insn following the brk.  */

			ex_vec = env->trap_vector;

			env->pregs[PR_ERP] = env->pc + 2;

			break;



		case EXCP_MMU_FAULT:

			ex_vec = env->fault_vector;

			env->pregs[PR_ERP] = env->pc;

			break;



		default:

			/* Is the core accepting interrupts?  */

			if (!(env->pregs[PR_CCS] & I_FLAG))

				return;

			/* The interrupt controller gives us the

			   vector.  */

			ex_vec = env->interrupt_vector;

			/* Normal interrupts are taken between

			   TB's.  env->pc is valid here.  */

			env->pregs[PR_ERP] = env->pc;

			break;

	}



	if ((env->pregs[PR_CCS] & U_FLAG)) {

		D(fprintf(logfile, "excp isr=%x PC=%x ERP=%x pid=%x ccs=%x cc=%d %x\n",

			  ex_vec, env->pc,

			  env->pregs[PR_ERP], env->pregs[PR_PID],

			  env->pregs[PR_CCS],

			  env->cc_op, env->cc_mask));

	}

	

	env->pc = ldl_code(env->pregs[PR_EBP] + ex_vec * 4);



	if (env->pregs[PR_CCS] & U_FLAG) {

		/* Swap stack pointers.  */

		env->pregs[PR_USP] = env->regs[R_SP];

		env->regs[R_SP] = env->ksp;

	}



	/* Apply the CRIS CCS shift. Clears U if set.  */

	cris_shift_ccs(env);

	D(fprintf (logfile, "%s isr=%x vec=%x ccs=%x pid=%d erp=%x\n", 

		   __func__, env->pc, ex_vec, 

		   env->pregs[PR_CCS],

		   env->pregs[PR_PID], 

		   env->pregs[PR_ERP]));

}
