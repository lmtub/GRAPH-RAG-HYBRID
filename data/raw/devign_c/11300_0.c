static unsigned int dec_move_r(DisasContext *dc)

{

	int size = memsize_zz(dc);



	DIS(fprintf (logfile, "move.%c $r%u, $r%u\n",

		    memsize_char(size), dc->op1, dc->op2));



	cris_cc_mask(dc, CC_MASK_NZ);

	if (size == 4) {

		dec_prep_move_r(dc, dc->op1, dc->op2, size, 0, cpu_R[dc->op2]);

		cris_cc_mask(dc, CC_MASK_NZ);

		cris_update_cc_op(dc, CC_OP_MOVE, 4);

		cris_update_cc_x(dc);

		cris_update_result(dc, cpu_R[dc->op2]);

	}

	else {

		TCGv t0;



		t0 = tcg_temp_new(TCG_TYPE_TL);

		dec_prep_move_r(dc, dc->op1, dc->op2, size, 0, t0);

		cris_alu(dc, CC_OP_MOVE,

			 cpu_R[dc->op2],

			 cpu_R[dc->op2], t0, size);

		tcg_temp_free(t0);

	}

	return 2;

}
