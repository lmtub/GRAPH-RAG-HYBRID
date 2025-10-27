static unsigned int dec_move_pm(DisasContext *dc)

{

	TCGv t0;

	int memsize;



	memsize = preg_sizes[dc->op2];



	DIS(fprintf (logfile, "move.%c $p%u, [$r%u%s\n",

		     memsize_char(memsize), 

		     dc->op2, dc->op1, dc->postinc ? "+]" : "]"));



	/* prepare store. Address in T0, value in T1.  */

	if (dc->op2 == PR_CCS)

		cris_evaluate_flags(dc);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	t_gen_mov_TN_preg(t0, dc->op2);

	cris_flush_cc_state(dc);

	gen_store(dc, cpu_R[dc->op1], t0, memsize);

	tcg_temp_free(t0);



	cris_cc_mask(dc, 0);

	if (dc->postinc)

		tcg_gen_addi_tl(cpu_R[dc->op1], cpu_R[dc->op1], memsize);

	return 2;

}
