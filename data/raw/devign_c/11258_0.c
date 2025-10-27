static void spr_read_sdr1 (void *opaque, int gprn, int sprn)

{

    tcg_gen_ld_tl(cpu_gpr[gprn], cpu_env, offsetof(CPUState, sdr1));

}
