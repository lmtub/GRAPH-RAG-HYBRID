static inline void _t_gen_mov_env_TN(int offset, TCGv tn)

{

    if (offset > sizeof(CPUCRISState)) {

        fprintf(stderr, "wrong store to env at off=%d\n", offset);

    }

    tcg_gen_st_tl(tn, cpu_env, offset);

}
