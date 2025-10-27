static int spapr_fixup_cpu_dt(void *fdt, sPAPRMachineState *spapr)

{

    int ret = 0, offset, cpus_offset;

    CPUState *cs;

    char cpu_model[32];

    int smt = kvmppc_smt_threads();

    uint32_t pft_size_prop[] = {0, cpu_to_be32(spapr->htab_shift)};



    CPU_FOREACH(cs) {

        PowerPCCPU *cpu = POWERPC_CPU(cs);

        DeviceClass *dc = DEVICE_GET_CLASS(cs);

        int index = spapr_vcpu_id(cpu);

        int compat_smt = MIN(smp_threads, ppc_compat_max_threads(cpu));



        if ((index % smt) != 0) {

            continue;

        }



        snprintf(cpu_model, 32, "%s@%x", dc->fw_name, index);



        cpus_offset = fdt_path_offset(fdt, "/cpus");

        if (cpus_offset < 0) {

            cpus_offset = fdt_add_subnode(fdt, 0, "cpus");

            if (cpus_offset < 0) {

                return cpus_offset;

            }

        }

        offset = fdt_subnode_offset(fdt, cpus_offset, cpu_model);

        if (offset < 0) {

            offset = fdt_add_subnode(fdt, cpus_offset, cpu_model);

            if (offset < 0) {

                return offset;

            }

        }



        ret = fdt_setprop(fdt, offset, "ibm,pft-size",

                          pft_size_prop, sizeof(pft_size_prop));

        if (ret < 0) {

            return ret;

        }



        if (nb_numa_nodes > 1) {

            ret = spapr_fixup_cpu_numa_dt(fdt, offset, cpu);

            if (ret < 0) {

                return ret;

            }

        }



        ret = spapr_fixup_cpu_smt_dt(fdt, offset, cpu, compat_smt);

        if (ret < 0) {

            return ret;

        }



        spapr_populate_pa_features(cpu, fdt, offset,

                                         spapr->cas_legacy_guest_workaround);

    }

    return ret;

}
