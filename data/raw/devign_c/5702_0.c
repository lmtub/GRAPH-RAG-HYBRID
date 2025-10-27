static unsigned int dec_movem_mr(DisasContext *dc)

{

	TCGv tmp[16];

	TCGv addr;

	int i;

	int nr = dc->op2 + 1;



	DIS(fprintf (logfile, "movem [$r%u%s, $r%u\n", dc->op1,

		    dc->postinc ? "+]" : "]", dc->op2));



	addr = tcg_temp_new(TCG_TYPE_TL);

	/* There are probably better ways of doing this.  */

	cris_flush_cc_state(dc);

	for (i = 0; i < (nr >> 1); i++) {

		tmp[i] = tcg_temp_new(TCG_TYPE_I64);

		tcg_gen_addi_tl(addr, cpu_R[dc->op1], i * 8);

		gen_load(dc, tmp[i], addr, 8, 0);

	}

	if (nr & 1) {

		tmp[i] = tcg_temp_new(TCG_TYPE_I32);

		tcg_gen_addi_tl(addr, cpu_R[dc->op1], i * 8);

		gen_load(dc, tmp[i], addr, 4, 0);

	}

	tcg_temp_free(addr);



	for (i = 0; i < (nr >> 1); i++) {

		tcg_gen_trunc_i64_i32(cpu_R[i * 2], tmp[i]);

		tcg_gen_shri_i64(tmp[i], tmp[i], 32);

		tcg_gen_trunc_i64_i32(cpu_R[i * 2 + 1], tmp[i]);

		tcg_temp_free(tmp[i]);

	}

	if (nr & 1) {

		tcg_gen_mov_tl(cpu_R[dc->op2], tmp[i]);

		tcg_temp_free(tmp[i]);

	}



	/* writeback the updated pointer value.  */

	if (dc->postinc)

		tcg_gen_addi_tl(cpu_R[dc->op1], cpu_R[dc->op1], nr * 4);



	/* gen_load might want to evaluate the previous insns flags.  */

	cris_cc_mask(dc, 0);

	return 2;

}
