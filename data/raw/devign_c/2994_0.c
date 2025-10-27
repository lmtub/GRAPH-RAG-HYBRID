static inline void t_gen_raise_exception(uint32_t index)

{

	tcg_gen_helper_0_1(helper_raise_exception, tcg_const_tl(index));

}
