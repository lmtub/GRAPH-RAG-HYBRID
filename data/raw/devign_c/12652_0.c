void list_cpus(FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

               const char *optarg)

{

    /* XXX: implement xxx_cpu_list for targets that still miss it */

#if defined(cpu_list_id)

    cpu_list_id(f, cpu_fprintf, optarg);

#elif defined(cpu_list)

    cpu_list(f, cpu_fprintf); /* deprecated */

#endif

}
