void cris_cpu_list(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    unsigned int i;



    (*cpu_fprintf)(f, "Available CPUs:\n");

    for (i = 0; i < ARRAY_SIZE(cris_cores); i++) {

        (*cpu_fprintf)(f, "  %s\n", cris_cores[i].name);

    }

}
