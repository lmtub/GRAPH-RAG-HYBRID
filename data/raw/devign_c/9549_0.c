void set_numa_modes(void)

{

    CPUArchState *env;

    int i;



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        for (i = 0; i < nb_numa_nodes; i++) {

            if (node_cpumask[i] & (1 << env->cpu_index)) {

                env->numa_node = i;

            }

        }

    }

}
