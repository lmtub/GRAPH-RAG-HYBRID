static inline void gen_bx_im(DisasContext *s, uint32_t addr)

{

    TCGv tmp;



    s->is_jmp = DISAS_UPDATE;

    if (s->thumb != (addr & 1)) {

        tmp = new_tmp();

        tcg_gen_movi_i32(tmp, addr & 1);

        tcg_gen_st_i32(tmp, cpu_env, offsetof(CPUState, thumb));

        dead_tmp(tmp);

    }

    tcg_gen_movi_i32(cpu_R[15], addr & ~1);

}
