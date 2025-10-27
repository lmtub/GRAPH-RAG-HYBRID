static unsigned int dec_move_mr(DisasContext *dc)

{

	int memsize = memsize_zz(dc);

	int insn_len;

	DIS(fprintf (logfile, "move.%c [$r%u%s, $r%u\n",

		    memsize_char(memsize),

		    dc->op1, dc->postinc ? "+]" : "]",

		    dc->op2));



	if (memsize == 4) {

		insn_len = dec_prep_move_m(dc, 0, 4, cpu_R[dc->op2]);

		cris_cc_mask(dc, CC_MASK_NZ);

		cris_update_cc_op(dc, CC_OP_MOVE, 4);

		cris_update_cc_x(dc);

		cris_update_result(dc, cpu_R[dc->op2]);

	}

	else {

		TCGv t0;



		t0 = tcg_temp_new(TCG_TYPE_TL);

		insn_len = dec_prep_move_m(dc, 0, memsize, t0);

		cris_cc_mask(dc, CC_MASK_NZ);

		cris_alu(dc, CC_OP_MOVE,

			    cpu_R[dc->op2], cpu_R[dc->op2], t0, memsize);

		tcg_temp_free(t0);

	}

	do_postinc(dc, memsize);

	return insn_len;

}
