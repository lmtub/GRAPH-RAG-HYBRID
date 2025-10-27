static unsigned int dec_btst_r(DisasContext *dc)

{

	TCGv l0;

	DIS(fprintf (logfile, "btst $r%u, $r%u\n",

		    dc->op1, dc->op2));

	cris_cc_mask(dc, CC_MASK_NZ);



	l0 = tcg_temp_local_new(TCG_TYPE_TL);

	cris_alu(dc, CC_OP_BTST, l0, cpu_R[dc->op2], cpu_R[dc->op1], 4);

	cris_update_cc_op(dc, CC_OP_FLAGS, 4);

	t_gen_mov_preg_TN(dc, PR_CCS, l0);

	dc->flags_uptodate = 1;

	tcg_temp_free(l0);

	return 2;

}
