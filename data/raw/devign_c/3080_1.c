static int spapr_fixup_cpu_smt_dt(void *fdt, int offset, PowerPCCPU *cpu,

                                  int smt_threads)

{

    int i, ret = 0;

    uint32_t servers_prop[smt_threads];

    uint32_t gservers_prop[smt_threads * 2];

    int index = ppc_get_vcpu_dt_id(cpu);



    if (cpu->cpu_version) {

        ret = fdt_setprop(fdt, offset, "cpu-version",

                          &cpu->cpu_version, sizeof(cpu->cpu_version));

        if (ret < 0) {

            return ret;

        }

    }



    /* Build interrupt servers and gservers properties */

    for (i = 0; i < smt_threads; i++) {

        servers_prop[i] = cpu_to_be32(index + i);

        /* Hack, direct the group queues back to cpu 0 */

        gservers_prop[i*2] = cpu_to_be32(index + i);

        gservers_prop[i*2 + 1] = 0;

    }

    ret = fdt_setprop(fdt, offset, "ibm,ppc-interrupt-server#s",

                      servers_prop, sizeof(servers_prop));

    if (ret < 0) {

        return ret;

    }

    ret = fdt_setprop(fdt, offset, "ibm,ppc-interrupt-gserver#s",

                      gservers_prop, sizeof(gservers_prop));



    return ret;

}
