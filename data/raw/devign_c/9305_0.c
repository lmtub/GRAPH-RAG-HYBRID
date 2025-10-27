cpu_mips_check_sign_extensions (CPUState *env, FILE *f,

                                int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                                int flags)

{

    int i;



    if (!SIGN_EXT_P(env->active_tc.PC))

        cpu_fprintf(f, "BROKEN: pc=0x" TARGET_FMT_lx "\n", env->active_tc.PC);

    if (!SIGN_EXT_P(env->active_tc.HI[0]))

        cpu_fprintf(f, "BROKEN: HI=0x" TARGET_FMT_lx "\n", env->active_tc.HI[0]);

    if (!SIGN_EXT_P(env->active_tc.LO[0]))

        cpu_fprintf(f, "BROKEN: LO=0x" TARGET_FMT_lx "\n", env->active_tc.LO[0]);

    if (!SIGN_EXT_P(env->btarget))

        cpu_fprintf(f, "BROKEN: btarget=0x" TARGET_FMT_lx "\n", env->btarget);



    for (i = 0; i < 32; i++) {

        if (!SIGN_EXT_P(env->active_tc.gpr[i]))

            cpu_fprintf(f, "BROKEN: %s=0x" TARGET_FMT_lx "\n", regnames[i], env->active_tc.gpr[i]);

    }



    if (!SIGN_EXT_P(env->CP0_EPC))

        cpu_fprintf(f, "BROKEN: EPC=0x" TARGET_FMT_lx "\n", env->CP0_EPC);

    if (!SIGN_EXT_P(env->lladdr))

        cpu_fprintf(f, "BROKEN: LLAddr=0x" TARGET_FMT_lx "\n", env->lladdr);

}
