static inline void gen_op_addq_ESP_im(int32_t val)

{

    tcg_gen_ld_tl(cpu_tmp0, cpu_env, offsetof(CPUState, regs[R_ESP]));

    tcg_gen_addi_tl(cpu_tmp0, cpu_tmp0, val);

    tcg_gen_st_tl(cpu_tmp0, cpu_env, offsetof(CPUState, regs[R_ESP]));

}
