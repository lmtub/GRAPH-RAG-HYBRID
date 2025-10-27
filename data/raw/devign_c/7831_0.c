void x86_cpu_list (FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    unsigned int i;



    for (i = 0; i < ARRAY_SIZE(x86_defs); i++)

        (*cpu_fprintf)(f, "x86 %16s\n", x86_defs[i].name);

}
