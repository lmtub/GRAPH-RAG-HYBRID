static void cris_alu(DisasContext *dc, int op,

			       TCGv d, TCGv op_a, TCGv op_b, int size)

{

	TCGv tmp;

	int writeback;



	writeback = 1;



	if (op == CC_OP_BOUND || op == CC_OP_BTST)

		tmp = tcg_temp_local_new(TCG_TYPE_TL);



	if (op == CC_OP_CMP) {

		tmp = tcg_temp_new(TCG_TYPE_TL);

		writeback = 0;

	} else if (size == 4) {

		tmp = d;

		writeback = 0;

	} else

		tmp = tcg_temp_new(TCG_TYPE_TL);





	cris_pre_alu_update_cc(dc, op, op_a, op_b, size);

	cris_alu_op_exec(dc, op, tmp, op_a, op_b, size);

	cris_update_result(dc, tmp);



	/* Writeback.  */

	if (writeback) {

		if (size == 1)

			tcg_gen_andi_tl(d, d, ~0xff);

		else

			tcg_gen_andi_tl(d, d, ~0xffff);

		tcg_gen_or_tl(d, d, tmp);

	}

	if (GET_TCGV(tmp) != GET_TCGV(d))

		tcg_temp_free(tmp);

}
