static unsigned int dec_swap_r(DisasContext *dc)

{

	TCGv t0;

#if DISAS_CRIS

	char modename[4];

#endif

	DIS(fprintf (logfile, "swap%s $r%u\n",

		     swapmode_name(dc->op2, modename), dc->op1));



	cris_cc_mask(dc, CC_MASK_NZ);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	t_gen_mov_TN_reg(t0, dc->op1);

	if (dc->op2 & 8)

		tcg_gen_not_tl(t0, t0);

	if (dc->op2 & 4)

		t_gen_swapw(t0, t0);

	if (dc->op2 & 2)

		t_gen_swapb(t0, t0);

	if (dc->op2 & 1)

		t_gen_swapr(t0, t0);

	cris_alu(dc, CC_OP_MOVE,

		    cpu_R[dc->op1], cpu_R[dc->op1], t0, 4);

	tcg_temp_free(t0);

	return 2;

}
