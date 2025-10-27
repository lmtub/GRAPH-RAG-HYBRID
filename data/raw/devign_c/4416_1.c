static int spapr_set_associativity(void *fdt, sPAPREnvironment *spapr)

{

    int ret = 0, offset;

    CPUPPCState *env;

    char cpu_model[32];

    int smt = kvmppc_smt_threads();



    assert(spapr->cpu_model);



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        uint32_t associativity[] = {cpu_to_be32(0x5),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(0x0),

                                    cpu_to_be32(env->numa_node),

                                    cpu_to_be32(env->cpu_index)};



        if ((env->cpu_index % smt) != 0) {

            continue;

        }



        snprintf(cpu_model, 32, "/cpus/%s@%x", spapr->cpu_model,

                 env->cpu_index);



        offset = fdt_path_offset(fdt, cpu_model);

        if (offset < 0) {

            return offset;

        }



        ret = fdt_setprop(fdt, offset, "ibm,associativity", associativity,

                          sizeof(associativity));

        if (ret < 0) {

            return ret;

        }

    }

    return ret;

}
