void arm_cpu_list(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    int i;



    (*cpu_fprintf)(f, "Available CPUs:\n");

    for (i = 0; arm_cpu_names[i].name; i++) {

        (*cpu_fprintf)(f, "  %s\n", arm_cpu_names[i].name);

    }

}
