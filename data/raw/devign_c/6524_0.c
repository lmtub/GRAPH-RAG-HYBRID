static void spapr_populate_cpu_dt(CPUState *cs, void *fdt, int offset,

                                  sPAPRMachineState *spapr)

{

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cs);

    int index = ppc_get_vcpu_dt_id(cpu);

    uint32_t segs[] = {cpu_to_be32(28), cpu_to_be32(40),

                       0xffffffff, 0xffffffff};

    uint32_t tbfreq = kvm_enabled() ? kvmppc_get_tbfreq()

        : SPAPR_TIMEBASE_FREQ;

    uint32_t cpufreq = kvm_enabled() ? kvmppc_get_clockfreq() : 1000000000;

    uint32_t page_sizes_prop[64];

    size_t page_sizes_prop_size;

    uint32_t vcpus_per_socket = smp_threads * smp_cores;

    uint32_t pft_size_prop[] = {0, cpu_to_be32(spapr->htab_shift)};

    int compat_smt = MIN(smp_threads, ppc_compat_max_threads(cpu));

    sPAPRDRConnector *drc;

    int drc_index;

    uint32_t radix_AP_encodings[PPC_PAGE_SIZES_MAX_SZ];

    int i;



    drc = spapr_drc_by_id(TYPE_SPAPR_DRC_CPU, index);

    if (drc) {

        drc_index = spapr_drc_index(drc);

        _FDT((fdt_setprop_cell(fdt, offset, "ibm,my-drc-index", drc_index)));

    }



    _FDT((fdt_setprop_cell(fdt, offset, "reg", index)));

    _FDT((fdt_setprop_string(fdt, offset, "device_type", "cpu")));



    _FDT((fdt_setprop_cell(fdt, offset, "cpu-version", env->spr[SPR_PVR])));

    _FDT((fdt_setprop_cell(fdt, offset, "d-cache-block-size",

                           env->dcache_line_size)));

    _FDT((fdt_setprop_cell(fdt, offset, "d-cache-line-size",

                           env->dcache_line_size)));

    _FDT((fdt_setprop_cell(fdt, offset, "i-cache-block-size",

                           env->icache_line_size)));

    _FDT((fdt_setprop_cell(fdt, offset, "i-cache-line-size",

                           env->icache_line_size)));



    if (pcc->l1_dcache_size) {

        _FDT((fdt_setprop_cell(fdt, offset, "d-cache-size",

                               pcc->l1_dcache_size)));

    } else {

        error_report("Warning: Unknown L1 dcache size for cpu");

    }

    if (pcc->l1_icache_size) {

        _FDT((fdt_setprop_cell(fdt, offset, "i-cache-size",

                               pcc->l1_icache_size)));

    } else {

        error_report("Warning: Unknown L1 icache size for cpu");

    }



    _FDT((fdt_setprop_cell(fdt, offset, "timebase-frequency", tbfreq)));

    _FDT((fdt_setprop_cell(fdt, offset, "clock-frequency", cpufreq)));

    _FDT((fdt_setprop_cell(fdt, offset, "slb-size", env->slb_nr)));

    _FDT((fdt_setprop_cell(fdt, offset, "ibm,slb-size", env->slb_nr)));

    _FDT((fdt_setprop_string(fdt, offset, "status", "okay")));

    _FDT((fdt_setprop(fdt, offset, "64-bit", NULL, 0)));



    if (env->spr_cb[SPR_PURR].oea_read) {

        _FDT((fdt_setprop(fdt, offset, "ibm,purr", NULL, 0)));

    }



    if (env->mmu_model & POWERPC_MMU_1TSEG) {

        _FDT((fdt_setprop(fdt, offset, "ibm,processor-segment-sizes",

                          segs, sizeof(segs))));

    }



    /* Advertise VMX/VSX (vector extensions) if available

     *   0 / no property == no vector extensions

     *   1               == VMX / Altivec available

     *   2               == VSX available */

    if (env->insns_flags & PPC_ALTIVEC) {

        uint32_t vmx = (env->insns_flags2 & PPC2_VSX) ? 2 : 1;



        _FDT((fdt_setprop_cell(fdt, offset, "ibm,vmx", vmx)));

    }



    /* Advertise DFP (Decimal Floating Point) if available

     *   0 / no property == no DFP

     *   1               == DFP available */

    if (env->insns_flags2 & PPC2_DFP) {

        _FDT((fdt_setprop_cell(fdt, offset, "ibm,dfp", 1)));

    }



    page_sizes_prop_size = ppc_create_page_sizes_prop(env, page_sizes_prop,

                                                  sizeof(page_sizes_prop));

    if (page_sizes_prop_size) {

        _FDT((fdt_setprop(fdt, offset, "ibm,segment-page-sizes",

                          page_sizes_prop, page_sizes_prop_size)));

    }



    spapr_populate_pa_features(env, fdt, offset, false);



    _FDT((fdt_setprop_cell(fdt, offset, "ibm,chip-id",

                           cs->cpu_index / vcpus_per_socket)));



    _FDT((fdt_setprop(fdt, offset, "ibm,pft-size",

                      pft_size_prop, sizeof(pft_size_prop))));



    if (nb_numa_nodes > 1) {

        _FDT(spapr_fixup_cpu_numa_dt(fdt, offset, cpu));

    }



    _FDT(spapr_fixup_cpu_smt_dt(fdt, offset, cpu, compat_smt));



    if (pcc->radix_page_info) {

        for (i = 0; i < pcc->radix_page_info->count; i++) {

            radix_AP_encodings[i] =

                cpu_to_be32(pcc->radix_page_info->entries[i]);

        }

        _FDT((fdt_setprop(fdt, offset, "ibm,processor-radix-AP-encodings",

                          radix_AP_encodings,

                          pcc->radix_page_info->count *

                          sizeof(radix_AP_encodings[0]))));

    }

}
