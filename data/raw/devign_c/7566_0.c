static void cris_evaluate_flags(DisasContext *dc)

{

	if (!dc->flags_uptodate) {

		cris_flush_cc_state(dc);



		switch (dc->cc_op)

		{

			case CC_OP_MCP:

				gen_helper_evaluate_flags_mcp();

				break;

			case CC_OP_MULS:

				gen_helper_evaluate_flags_muls();

				break;

			case CC_OP_MULU:

				gen_helper_evaluate_flags_mulu();

				break;

			case CC_OP_MOVE:

			case CC_OP_AND:

			case CC_OP_OR:

			case CC_OP_XOR:

			case CC_OP_ASR:

			case CC_OP_LSR:

			case CC_OP_LSL:

				switch (dc->cc_size)

				{

					case 4:

						gen_helper_evaluate_flags_move_4();

						break;

					case 2:

						gen_helper_evaluate_flags_move_2();

						break;

					default:

						gen_helper_evaluate_flags();

						break;

				}

				break;

			case CC_OP_FLAGS:

				/* live.  */

				break;

			default:

			{

				switch (dc->cc_size)

				{

					case 4:

						gen_helper_evaluate_flags_alu_4();

						break;

					default:

						gen_helper_evaluate_flags();

						break;

				}

			}

			break;

		}

		if (dc->flagx_known) {

			if (dc->flags_x)

				tcg_gen_ori_tl(cpu_PR[PR_CCS], 

					       cpu_PR[PR_CCS], X_FLAG);

			else

				tcg_gen_andi_tl(cpu_PR[PR_CCS], 

						cpu_PR[PR_CCS], ~X_FLAG);

	        }



		dc->flags_uptodate = 1;

	}

}
