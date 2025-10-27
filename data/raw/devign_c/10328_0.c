void  helper_evaluate_flags_alu_4(void)

{

	uint32_t src;

	uint32_t dst;

	uint32_t res;

	uint32_t flags = 0;



	src = env->cc_src;

	dst = env->cc_dest;



	/* Reconstruct the result.  */

	switch (env->cc_op)

	{

		case CC_OP_SUB:

			res = dst - src;

			break;

		case CC_OP_ADD:

			res = dst + src;

			break;

		default:

			res = env->cc_result;

			break;

	}



	if (env->cc_op == CC_OP_SUB || env->cc_op == CC_OP_CMP)

		src = ~src;



	if ((res & 0x80000000L) != 0L)

	{

		flags |= N_FLAG;

		if (((src & 0x80000000L) == 0L)

		    && ((dst & 0x80000000L) == 0L))

		{

			flags |= V_FLAG;

		}

		else if (((src & 0x80000000L) != 0L) &&

			 ((dst & 0x80000000L) != 0L))

		{

			flags |= C_FLAG;

		}

	}

	else

	{

		if (res == 0L)

			flags |= Z_FLAG;

		if (((src & 0x80000000L) != 0L)

		    && ((dst & 0x80000000L) != 0L))

			flags |= V_FLAG;

		if ((dst & 0x80000000L) != 0L

		    || (src & 0x80000000L) != 0L)

			flags |= C_FLAG;

	}



	if (env->cc_op == CC_OP_SUB

	    || env->cc_op == CC_OP_CMP) {

		flags ^= C_FLAG;

	}

	evaluate_flags_writeback(flags);

}
