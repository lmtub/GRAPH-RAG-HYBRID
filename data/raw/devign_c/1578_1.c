static void *spapr_create_fdt_skel(const char *cpu_model,

                                   hwaddr initrd_base,

                                   hwaddr initrd_size,

                                   hwaddr kernel_size,

                                   const char *boot_device,

                                   const char *kernel_cmdline,

                                   uint32_t epow_irq)

{

    void *fdt;

    CPUPPCState *env;

    uint32_t start_prop = cpu_to_be32(initrd_base);

    uint32_t end_prop = cpu_to_be32(initrd_base + initrd_size);

    char hypertas_prop[] = "hcall-pft\0hcall-term\0hcall-dabr\0hcall-interrupt"

        "\0hcall-tce\0hcall-vio\0hcall-splpar\0hcall-bulk";

    char qemu_hypertas_prop[] = "hcall-memop1";

    uint32_t refpoints[] = {cpu_to_be32(0x4), cpu_to_be32(0x4)};

    uint32_t interrupt_server_ranges_prop[] = {0, cpu_to_be32(smp_cpus)};

    char *modelname;

    int i, smt = kvmppc_smt_threads();

    unsigned char vec5[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x80};



    fdt = g_malloc0(FDT_MAX_SIZE);

    _FDT((fdt_create(fdt, FDT_MAX_SIZE)));



    if (kernel_size) {

        _FDT((fdt_add_reservemap_entry(fdt, KERNEL_LOAD_ADDR, kernel_size)));

    }

    if (initrd_size) {

        _FDT((fdt_add_reservemap_entry(fdt, initrd_base, initrd_size)));

    }

    _FDT((fdt_finish_reservemap(fdt)));



    /* Root node */

    _FDT((fdt_begin_node(fdt, "")));

    _FDT((fdt_property_string(fdt, "device_type", "chrp")));

    _FDT((fdt_property_string(fdt, "model", "IBM pSeries (emulated by qemu)")));



    _FDT((fdt_property_cell(fdt, "#address-cells", 0x2)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x2)));



    /* /chosen */

    _FDT((fdt_begin_node(fdt, "chosen")));



    /* Set Form1_affinity */

    _FDT((fdt_property(fdt, "ibm,architecture-vec-5", vec5, sizeof(vec5))));



    _FDT((fdt_property_string(fdt, "bootargs", kernel_cmdline)));

    _FDT((fdt_property(fdt, "linux,initrd-start",

                       &start_prop, sizeof(start_prop))));

    _FDT((fdt_property(fdt, "linux,initrd-end",

                       &end_prop, sizeof(end_prop))));

    if (kernel_size) {

        uint64_t kprop[2] = { cpu_to_be64(KERNEL_LOAD_ADDR),

                              cpu_to_be64(kernel_size) };



        _FDT((fdt_property(fdt, "qemu,boot-kernel", &kprop, sizeof(kprop))));

    }

    _FDT((fdt_property_string(fdt, "qemu,boot-device", boot_device)));

    _FDT((fdt_property_cell(fdt, "qemu,graphic-width", graphic_width)));

    _FDT((fdt_property_cell(fdt, "qemu,graphic-height", graphic_height)));

    _FDT((fdt_property_cell(fdt, "qemu,graphic-depth", graphic_depth)));



    _FDT((fdt_end_node(fdt)));



    /* cpus */

    _FDT((fdt_begin_node(fdt, "cpus")));



    _FDT((fdt_property_cell(fdt, "#address-cells", 0x1)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x0)));



    modelname = g_strdup(cpu_model);



    for (i = 0; i < strlen(modelname); i++) {

        modelname[i] = toupper(modelname[i]);

    }



    /* This is needed during FDT finalization */

    spapr->cpu_model = g_strdup(modelname);



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        CPUState *cpu = CPU(ppc_env_get_cpu(env));

        int index = cpu->cpu_index;

        uint32_t servers_prop[smp_threads];

        uint32_t gservers_prop[smp_threads * 2];

        char *nodename;

        uint32_t segs[] = {cpu_to_be32(28), cpu_to_be32(40),

                           0xffffffff, 0xffffffff};

        uint32_t tbfreq = kvm_enabled() ? kvmppc_get_tbfreq() : TIMEBASE_FREQ;

        uint32_t cpufreq = kvm_enabled() ? kvmppc_get_clockfreq() : 1000000000;

        uint32_t page_sizes_prop[64];

        size_t page_sizes_prop_size;



        if ((index % smt) != 0) {

            continue;

        }



        if (asprintf(&nodename, "%s@%x", modelname, index) < 0) {

            fprintf(stderr, "Allocation failure\n");

            exit(1);

        }



        _FDT((fdt_begin_node(fdt, nodename)));



        free(nodename);



        _FDT((fdt_property_cell(fdt, "reg", index)));

        _FDT((fdt_property_string(fdt, "device_type", "cpu")));



        _FDT((fdt_property_cell(fdt, "cpu-version", env->spr[SPR_PVR])));

        _FDT((fdt_property_cell(fdt, "dcache-block-size",

                                env->dcache_line_size)));

        _FDT((fdt_property_cell(fdt, "icache-block-size",

                                env->icache_line_size)));

        _FDT((fdt_property_cell(fdt, "timebase-frequency", tbfreq)));

        _FDT((fdt_property_cell(fdt, "clock-frequency", cpufreq)));

        _FDT((fdt_property_cell(fdt, "ibm,slb-size", env->slb_nr)));

        _FDT((fdt_property_string(fdt, "status", "okay")));

        _FDT((fdt_property(fdt, "64-bit", NULL, 0)));



        /* Build interrupt servers and gservers properties */

        for (i = 0; i < smp_threads; i++) {

            servers_prop[i] = cpu_to_be32(index + i);

            /* Hack, direct the group queues back to cpu 0 */

            gservers_prop[i*2] = cpu_to_be32(index + i);

            gservers_prop[i*2 + 1] = 0;

        }

        _FDT((fdt_property(fdt, "ibm,ppc-interrupt-server#s",

                           servers_prop, sizeof(servers_prop))));

        _FDT((fdt_property(fdt, "ibm,ppc-interrupt-gserver#s",

                           gservers_prop, sizeof(gservers_prop))));



        if (env->mmu_model & POWERPC_MMU_1TSEG) {

            _FDT((fdt_property(fdt, "ibm,processor-segment-sizes",

                               segs, sizeof(segs))));

        }



        /* Advertise VMX/VSX (vector extensions) if available

         *   0 / no property == no vector extensions

         *   1               == VMX / Altivec available

         *   2               == VSX available */

        if (env->insns_flags & PPC_ALTIVEC) {

            uint32_t vmx = (env->insns_flags2 & PPC2_VSX) ? 2 : 1;



            _FDT((fdt_property_cell(fdt, "ibm,vmx", vmx)));

        }



        /* Advertise DFP (Decimal Floating Point) if available

         *   0 / no property == no DFP

         *   1               == DFP available */

        if (env->insns_flags2 & PPC2_DFP) {

            _FDT((fdt_property_cell(fdt, "ibm,dfp", 1)));

        }



        page_sizes_prop_size = create_page_sizes_prop(env, page_sizes_prop,

                                                      sizeof(page_sizes_prop));

        if (page_sizes_prop_size) {

            _FDT((fdt_property(fdt, "ibm,segment-page-sizes",

                               page_sizes_prop, page_sizes_prop_size)));

        }



        _FDT((fdt_end_node(fdt)));

    }



    g_free(modelname);



    _FDT((fdt_end_node(fdt)));



    /* RTAS */

    _FDT((fdt_begin_node(fdt, "rtas")));



    _FDT((fdt_property(fdt, "ibm,hypertas-functions", hypertas_prop,

                       sizeof(hypertas_prop))));

    _FDT((fdt_property(fdt, "qemu,hypertas-functions", qemu_hypertas_prop,

                       sizeof(qemu_hypertas_prop))));



    _FDT((fdt_property(fdt, "ibm,associativity-reference-points",

        refpoints, sizeof(refpoints))));



    _FDT((fdt_property_cell(fdt, "rtas-error-log-max", RTAS_ERROR_LOG_MAX)));



    _FDT((fdt_end_node(fdt)));



    /* interrupt controller */

    _FDT((fdt_begin_node(fdt, "interrupt-controller")));



    _FDT((fdt_property_string(fdt, "device_type",

                              "PowerPC-External-Interrupt-Presentation")));

    _FDT((fdt_property_string(fdt, "compatible", "IBM,ppc-xicp")));

    _FDT((fdt_property(fdt, "interrupt-controller", NULL, 0)));

    _FDT((fdt_property(fdt, "ibm,interrupt-server-ranges",

                       interrupt_server_ranges_prop,

                       sizeof(interrupt_server_ranges_prop))));

    _FDT((fdt_property_cell(fdt, "#interrupt-cells", 2)));

    _FDT((fdt_property_cell(fdt, "linux,phandle", PHANDLE_XICP)));

    _FDT((fdt_property_cell(fdt, "phandle", PHANDLE_XICP)));



    _FDT((fdt_end_node(fdt)));



    /* vdevice */

    _FDT((fdt_begin_node(fdt, "vdevice")));



    _FDT((fdt_property_string(fdt, "device_type", "vdevice")));

    _FDT((fdt_property_string(fdt, "compatible", "IBM,vdevice")));

    _FDT((fdt_property_cell(fdt, "#address-cells", 0x1)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x0)));

    _FDT((fdt_property_cell(fdt, "#interrupt-cells", 0x2)));

    _FDT((fdt_property(fdt, "interrupt-controller", NULL, 0)));



    _FDT((fdt_end_node(fdt)));



    /* event-sources */

    spapr_events_fdt_skel(fdt, epow_irq);



    _FDT((fdt_end_node(fdt))); /* close root node */

    _FDT((fdt_finish(fdt)));



    return fdt;

}
