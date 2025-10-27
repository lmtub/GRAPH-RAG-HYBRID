static inline void t_gen_swapr(TCGv d, TCGv s)

{

	struct {

		int shift; /* LSL when positive, LSR when negative.  */

		uint32_t mask;

	} bitrev [] = {

		{7, 0x80808080},

		{5, 0x40404040},

		{3, 0x20202020},

		{1, 0x10101010},

		{-1, 0x08080808},

		{-3, 0x04040404},

		{-5, 0x02020202},

		{-7, 0x01010101}

	};

	int i;

	TCGv t, org_s;



	/* d and s refer the same object.  */

	t = tcg_temp_new(TCG_TYPE_TL);

	org_s = tcg_temp_new(TCG_TYPE_TL);

	tcg_gen_mov_tl(org_s, s);



	tcg_gen_shli_tl(t, org_s,  bitrev[0].shift);

	tcg_gen_andi_tl(d, t,  bitrev[0].mask);

	for (i = 1; i < sizeof bitrev / sizeof bitrev[0]; i++) {

		if (bitrev[i].shift >= 0) {

			tcg_gen_shli_tl(t, org_s,  bitrev[i].shift);

		} else {

			tcg_gen_shri_tl(t, org_s,  -bitrev[i].shift);

		}

		tcg_gen_andi_tl(t, t,  bitrev[i].mask);

		tcg_gen_or_tl(d, d, t);

	}

	tcg_temp_free(t);

	tcg_temp_free(org_s);

}
