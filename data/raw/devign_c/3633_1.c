static inline void gen_bx(DisasContext *s, TCGv_i32 var)

{

    s->is_jmp = DISAS_UPDATE;

    tcg_gen_andi_i32(cpu_R[15], var, ~1);

    tcg_gen_andi_i32(var, var, 1);

    store_cpu_field(var, thumb);

}
