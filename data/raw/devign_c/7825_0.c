static inline void cris_alu_alloc_temps(DisasContext *dc, int size, TCGv *t)

{

	if (size == 4) {

		t[0] = cpu_R[dc->op2];

		t[1] = cpu_R[dc->op1];

	} else {

		t[0] = tcg_temp_new(TCG_TYPE_TL);

		t[1] = tcg_temp_new(TCG_TYPE_TL);

	}

}
