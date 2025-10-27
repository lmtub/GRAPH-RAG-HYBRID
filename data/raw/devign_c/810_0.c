static unsigned int dec_move_rp(DisasContext *dc)

{

	TCGv t[2];

	DIS(fprintf (logfile, "move $r%u, $p%u\n", dc->op1, dc->op2));

	cris_cc_mask(dc, 0);



	t[0] = tcg_temp_new(TCG_TYPE_TL);

	if (dc->op2 == PR_CCS) {

		cris_evaluate_flags(dc);

		t_gen_mov_TN_reg(t[0], dc->op1);

		if (dc->tb_flags & U_FLAG) {

			t[1] = tcg_temp_new(TCG_TYPE_TL);

			/* User space is not allowed to touch all flags.  */

			tcg_gen_andi_tl(t[0], t[0], 0x39f);

			tcg_gen_andi_tl(t[1], cpu_PR[PR_CCS], ~0x39f);

			tcg_gen_or_tl(t[0], t[1], t[0]);

			tcg_temp_free(t[1]);

		}

	}

	else

		t_gen_mov_TN_reg(t[0], dc->op1);



	t_gen_mov_preg_TN(dc, dc->op2, t[0]);

	if (dc->op2 == PR_CCS) {

		cris_update_cc_op(dc, CC_OP_FLAGS, 4);

		dc->flags_uptodate = 1;

	}

	tcg_temp_free(t[0]);

	return 2;

}
