static inline void cris_update_result(DisasContext *dc, TCGv res)

{

	if (dc->update_cc) {

		if (dc->cc_size == 4 && 

		    (dc->cc_op == CC_OP_SUB

		     || dc->cc_op == CC_OP_ADD))

			return;

		tcg_gen_mov_tl(cc_result, res);

	}

}
