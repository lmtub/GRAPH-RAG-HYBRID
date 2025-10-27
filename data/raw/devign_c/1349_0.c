static inline void gen_op_addl_A0_seg(DisasContext *s, int reg)

{

    tcg_gen_ld_tl(cpu_tmp0, cpu_env, offsetof(CPUX86State, segs[reg].base));

    if (CODE64(s)) {

        tcg_gen_ext32u_tl(cpu_A0, cpu_A0);

        tcg_gen_add_tl(cpu_A0, cpu_A0, cpu_tmp0);

    } else {

        tcg_gen_add_tl(cpu_A0, cpu_A0, cpu_tmp0);

        tcg_gen_ext32u_tl(cpu_A0, cpu_A0);

    }

}
