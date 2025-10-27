static unsigned int dec_bound_r(DisasContext *dc)

{

	TCGv l0;

	int size = memsize_zz(dc);

	DIS(fprintf (logfile, "bound.%c $r%u, $r%u\n",

		    memsize_char(size), dc->op1, dc->op2));

	cris_cc_mask(dc, CC_MASK_NZ);

	l0 = tcg_temp_local_new(TCG_TYPE_TL);

	dec_prep_move_r(dc, dc->op1, dc->op2, size, 0, l0);

	cris_alu(dc, CC_OP_BOUND, cpu_R[dc->op2], cpu_R[dc->op2], l0, 4);

	tcg_temp_free(l0);

	return 2;

}
