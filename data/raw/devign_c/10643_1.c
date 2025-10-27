void cpu_dump_state(CPUState *env, FILE *f,

                    int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                    int flags)

{

    int i, x;



    cpu_fprintf(f, "pc: " TARGET_FMT_lx "  npc: " TARGET_FMT_lx "\n", env->pc,

                env->npc);

    cpu_fprintf(f, "General Registers:\n");

    for (i = 0; i < 4; i++)

        cpu_fprintf(f, "%%g%c: " TARGET_FMT_lx "\t", i + '0', env->gregs[i]);

    cpu_fprintf(f, "\n");

    for (; i < 8; i++)

        cpu_fprintf(f, "%%g%c: " TARGET_FMT_lx "\t", i + '0', env->gregs[i]);

    cpu_fprintf(f, "\nCurrent Register Window:\n");

    for (x = 0; x < 3; x++) {

        for (i = 0; i < 4; i++)

            cpu_fprintf(f, "%%%c%d: " TARGET_FMT_lx "\t",

                    (x == 0 ? 'o' : (x == 1 ? 'l' : 'i')), i,

                    env->regwptr[i + x * 8]);

        cpu_fprintf(f, "\n");

        for (; i < 8; i++)

            cpu_fprintf(f, "%%%c%d: " TARGET_FMT_lx "\t",

                    (x == 0 ? 'o' : x == 1 ? 'l' : 'i'), i,

                    env->regwptr[i + x * 8]);

        cpu_fprintf(f, "\n");

    }

    cpu_fprintf(f, "\nFloating Point Registers:\n");

    for (i = 0; i < 32; i++) {

        if ((i & 3) == 0)

            cpu_fprintf(f, "%%f%02d:", i);

        cpu_fprintf(f, " %016lf", env->fpr[i]);

        if ((i & 3) == 3)

            cpu_fprintf(f, "\n");

    }

#ifdef TARGET_SPARC64

    cpu_fprintf(f, "pstate: 0x%08x ccr: 0x%02x asi: 0x%02x tl: %d fprs: %d\n",

                env->pstate, GET_CCR(env), env->asi, env->tl, env->fprs);

    cpu_fprintf(f, "cansave: %d canrestore: %d otherwin: %d wstate %d "

                "cleanwin %d cwp %d\n",

                env->cansave, env->canrestore, env->otherwin, env->wstate,

                env->cleanwin, env->nwindows - 1 - env->cwp);

#else

    cpu_fprintf(f, "psr: 0x%08x -> %c%c%c%c %c%c%c wim: 0x%08x\n",

                GET_PSR(env), GET_FLAG(PSR_ZERO, 'Z'), GET_FLAG(PSR_OVF, 'V'),

                GET_FLAG(PSR_NEG, 'N'), GET_FLAG(PSR_CARRY, 'C'),

                env->psrs?'S':'-', env->psrps?'P':'-',

                env->psret?'E':'-', env->wim);

#endif

    cpu_fprintf(f, "fsr: 0x%08x\n", GET_FSR32(env));

}
