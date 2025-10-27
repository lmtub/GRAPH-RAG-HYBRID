void sh4_cpu_list(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    int i;



    for (i = 0; i < ARRAY_SIZE(sh4_defs); i++)

	(*cpu_fprintf)(f, "%s\n", sh4_defs[i].name);

}
