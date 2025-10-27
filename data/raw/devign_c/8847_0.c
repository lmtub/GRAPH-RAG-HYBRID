static void powernv_create_core_node(PnvChip *chip, PnvCore *pc, void *fdt)

{

    CPUState *cs = CPU(DEVICE(pc->threads));

    DeviceClass *dc = DEVICE_GET_CLASS(cs);

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    int smt_threads = CPU_CORE(pc)->nr_threads;

    CPUPPCState *env = &cpu->env;

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cs);

    uint32_t servers_prop[smt_threads];

    int i;

    uint32_t segs[] = {cpu_to_be32(28), cpu_to_be32(40),

                       0xffffffff, 0xffffffff};

    uint32_t tbfreq = PNV_TIMEBASE_FREQ;

    uint32_t cpufreq = 1000000000;

    uint32_t page_sizes_prop[64];

    size_t page_sizes_prop_size;

    const uint8_t pa_features[] = { 24, 0,

                                    0xf6, 0x3f, 0xc7, 0xc0, 0x80, 0xf0,

                                    0x80, 0x00, 0x00, 0x00, 0x00, 0x00,

                                    0x00, 0x00, 0x00, 0x00, 0x80, 0x00,

                                    0x80, 0x00, 0x80, 0x00, 0x80, 0x00 };

    int offset;

    char *nodename;

    int cpus_offset = get_cpus_node(fdt);



    nodename = g_strdup_printf("%s@%x", dc->fw_name, pc->pir);

    offset = fdt_add_subnode(fdt, cpus_offset, nodename);

    _FDT(offset);

    g_free(nodename);



    _FDT((fdt_setprop_cell(fdt, offset, "ibm,chip-id", chip->chip_id)));



    _FDT((fdt_setprop_cell(fdt, offset, "reg", pc->pir)));

    _FDT((fdt_setprop_cell(fdt, offset, "ibm,pir", pc->pir)));

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



    _FDT((fdt_setprop(fdt, offset, "ibm,pa-features",

                       pa_features, sizeof(pa_features))));



    /* Build interrupt servers properties */

    for (i = 0; i < smt_threads; i++) {

        servers_prop[i] = cpu_to_be32(pc->pir + i);

    }

    _FDT((fdt_setprop(fdt, offset, "ibm,ppc-interrupt-server#s",

                       servers_prop, sizeof(servers_prop))));

}
