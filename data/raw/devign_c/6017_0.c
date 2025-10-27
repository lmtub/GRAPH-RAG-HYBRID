void mips_cpu_list (FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    int i;



    for (i = 0; i < ARRAY_SIZE(mips_defs); i++) {

        (*cpu_fprintf)(f, "MIPS '%s'\n",

                       mips_defs[i].name);

    }

}
