static int gen_set_psr_im(DisasContext *s, uint32_t mask, int spsr, uint32_t val)

{

    TCGv tmp;

    tmp = new_tmp();

    tcg_gen_movi_i32(tmp, val);

    return gen_set_psr(s, mask, spsr, tmp);

}
