static unsigned int dec_addi_acr(DisasContext *dc)

{

	TCGv t0;

	DIS(fprintf (logfile, "addi.%c $r%u, $r%u, $acr\n",

		  memsize_char(memsize_zz(dc)), dc->op2, dc->op1));

	cris_cc_mask(dc, 0);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	tcg_gen_shl_tl(t0, cpu_R[dc->op2], tcg_const_tl(dc->zzsize));

	tcg_gen_add_tl(cpu_R[R_ACR], cpu_R[dc->op1], t0);



	return 2;

}
