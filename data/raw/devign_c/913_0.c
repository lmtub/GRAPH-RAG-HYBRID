static char *enumerate_cpus(unsigned long *cpus, int max_cpus)

{

    int cpu;

    bool first = true;

    GString *s = g_string_new(NULL);



    for (cpu = find_first_bit(cpus, max_cpus);

        cpu < max_cpus;

        cpu = find_next_bit(cpus, max_cpus, cpu + 1)) {

        g_string_append_printf(s, "%s%d", first ? "" : " ", cpu);

        first = false;

    }

    return g_string_free(s, FALSE);

}
