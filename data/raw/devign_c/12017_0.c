static unsigned int dec_lz_r(DisasContext *dc)

{

	TCGv t0;

	DIS(fprintf (logfile, "lz $r%u, $r%u\n",

		    dc->op1, dc->op2));

	cris_cc_mask(dc, CC_MASK_NZ);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	dec_prep_alu_r(dc, dc->op1, dc->op2, 4, 0, cpu_R[dc->op2], t0);

	cris_alu(dc, CC_OP_LZ, cpu_R[dc->op2], cpu_R[dc->op2], t0, 4);

	tcg_temp_free(t0);

	return 2;

}
