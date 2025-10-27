static inline TCGv load_reg(DisasContext *s, int reg)

{

    TCGv tmp = new_tmp();

    load_reg_var(s, tmp, reg);

    return tmp;

}
