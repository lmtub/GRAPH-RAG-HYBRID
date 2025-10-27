static inline void gen_lookup_tb(DisasContext *s)

{

    tcg_gen_movi_i32(cpu_R[15], s->pc & ~1);

    s->is_jmp = DISAS_UPDATE;

}
