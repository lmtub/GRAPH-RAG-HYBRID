static unsigned int dec_move_pr(DisasContext *dc)

{

	TCGv t0;

	DIS(fprintf (logfile, "move $p%u, $r%u\n", dc->op1, dc->op2));

	cris_cc_mask(dc, 0);



	if (dc->op2 == PR_CCS)

		cris_evaluate_flags(dc);



	t0 = tcg_temp_new(TCG_TYPE_TL);

	t_gen_mov_TN_preg(t0, dc->op2);

	cris_alu(dc, CC_OP_MOVE,

		 cpu_R[dc->op1], cpu_R[dc->op1], t0, preg_sizes[dc->op2]);

	tcg_temp_free(t0);

	return 2;

}
