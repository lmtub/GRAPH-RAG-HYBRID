void m68k_cpu_list(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    unsigned int i;



    for (i = 0; m68k_cpu_defs[i].name; i++) {

        (*cpu_fprintf)(f, "%s\n", m68k_cpu_defs[i].name);

    }

}
