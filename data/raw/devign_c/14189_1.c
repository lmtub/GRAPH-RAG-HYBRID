static void *spapr_create_fdt_skel(hwaddr initrd_base,
                                   hwaddr initrd_size,
                                   hwaddr kernel_size,
                                   bool little_endian,
                                   const char *boot_device,
                                   const char *kernel_cmdline,
                                   uint32_t epow_irq)
{
    void *fdt;
    CPUState *cs;
    uint32_t start_prop = cpu_to_be32(initrd_base);
    uint32_t end_prop = cpu_to_be32(initrd_base + initrd_size);
    GString *hypertas = g_string_sized_new(256);
    GString *qemu_hypertas = g_string_sized_new(256);
    uint32_t refpoints[] = {cpu_to_be32(0x4), cpu_to_be32(0x4)};
    uint32_t interrupt_server_ranges_prop[] = {0, cpu_to_be32(smp_cpus)};
    int smt = kvmppc_smt_threads();
    unsigned char vec5[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x80};
    QemuOpts *opts = qemu_opts_find(qemu_find_opts("smp-opts"), NULL);
    unsigned sockets = opts ? qemu_opt_get_number(opts, "sockets", 0) : 0;
    uint32_t cpus_per_socket = sockets ? (smp_cpus / sockets) : 1;
    add_str(hypertas, "hcall-pft");
    add_str(hypertas, "hcall-term");
    add_str(hypertas, "hcall-dabr");
    add_str(hypertas, "hcall-interrupt");
    add_str(hypertas, "hcall-tce");
    add_str(hypertas, "hcall-vio");
    add_str(hypertas, "hcall-splpar");
    add_str(hypertas, "hcall-bulk");
    add_str(hypertas, "hcall-set-mode");
    add_str(qemu_hypertas, "hcall-memop1");
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
    _FDT((fdt_property_string(fdt, "compatible", "qemu,pseries")));
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
        if (little_endian) {
            _FDT((fdt_property(fdt, "qemu,boot-kernel-le", NULL, 0)));
        }
    }
    if (boot_device) {
        _FDT((fdt_property_string(fdt, "qemu,boot-device", boot_device)));
    }
    if (boot_menu) {
        _FDT((fdt_property_cell(fdt, "qemu,boot-menu", boot_menu)));
    }
    _FDT((fdt_property_cell(fdt, "qemu,graphic-width", graphic_width)));
    _FDT((fdt_property_cell(fdt, "qemu,graphic-height", graphic_height)));
    _FDT((fdt_property_cell(fdt, "qemu,graphic-depth", graphic_depth)));
    _FDT((fdt_end_node(fdt)));
    /* cpus */
    _FDT((fdt_begin_node(fdt, "cpus")));
    _FDT((fdt_property_cell(fdt, "#address-cells", 0x1)));
    _FDT((fdt_property_cell(fdt, "#size-cells", 0x0)));
    CPU_FOREACH(cs) {
        PowerPCCPU *cpu = POWERPC_CPU(cs);
        CPUPPCState *env = &cpu->env;
        DeviceClass *dc = DEVICE_GET_CLASS(cs);
        PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cs);
        int index = ppc_get_vcpu_dt_id(cpu);
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
        nodename = g_strdup_printf("%s@%x", dc->fw_name, index);
        _FDT((fdt_begin_node(fdt, nodename)));
        g_free(nodename);
        _FDT((fdt_property_cell(fdt, "reg", index)));
        _FDT((fdt_property_string(fdt, "device_type", "cpu")));
        _FDT((fdt_property_cell(fdt, "cpu-version", env->spr[SPR_PVR])));
        _FDT((fdt_property_cell(fdt, "d-cache-block-size",
                                env->dcache_line_size)));
        _FDT((fdt_property_cell(fdt, "d-cache-line-size",
                                env->dcache_line_size)));
        _FDT((fdt_property_cell(fdt, "i-cache-block-size",
                                env->icache_line_size)));
        _FDT((fdt_property_cell(fdt, "i-cache-line-size",
                                env->icache_line_size)));
        if (pcc->l1_dcache_size) {
            _FDT((fdt_property_cell(fdt, "d-cache-size", pcc->l1_dcache_size)));
        } else {
            fprintf(stderr, "Warning: Unknown L1 dcache size for cpu\n");
        }
        if (pcc->l1_icache_size) {
            _FDT((fdt_property_cell(fdt, "i-cache-size", pcc->l1_icache_size)));
        } else {
            fprintf(stderr, "Warning: Unknown L1 icache size for cpu\n");
        }
        _FDT((fdt_property_cell(fdt, "timebase-frequency", tbfreq)));
        _FDT((fdt_property_cell(fdt, "clock-frequency", cpufreq)));
        _FDT((fdt_property_cell(fdt, "ibm,slb-size", env->slb_nr)));
        _FDT((fdt_property_string(fdt, "status", "okay")));
        _FDT((fdt_property(fdt, "64-bit", NULL, 0)));
        if (env->spr_cb[SPR_PURR].oea_read) {
            _FDT((fdt_property(fdt, "ibm,purr", NULL, 0)));
        }
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
        _FDT((fdt_property_cell(fdt, "ibm,chip-id",
                                cs->cpu_index / cpus_per_socket)));
        _FDT((fdt_end_node(fdt)));
    }
    _FDT((fdt_end_node(fdt)));
    /* RTAS */
    _FDT((fdt_begin_node(fdt, "rtas")));
    if (!kvm_enabled() || kvmppc_spapr_use_multitce()) {
        add_str(hypertas, "hcall-multi-tce");
    }
    _FDT((fdt_property(fdt, "ibm,hypertas-functions", hypertas->str,
                       hypertas->len)));
    g_string_free(hypertas, TRUE);
    _FDT((fdt_property(fdt, "qemu,hypertas-functions", qemu_hypertas->str,
                       qemu_hypertas->len)));
    g_string_free(qemu_hypertas, TRUE);
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
    /* /hypervisor node */
    if (kvm_enabled()) {
        uint8_t hypercall[16];
        /* indicate KVM hypercall interface */
        _FDT((fdt_begin_node(fdt, "hypervisor")));
        _FDT((fdt_property_string(fdt, "compatible", "linux,kvm")));
        if (kvmppc_has_cap_fixup_hcalls()) {
             * Older KVM versions with older guest kernels were broken with the
             * magic page, don't allow the guest to map it.
            kvmppc_get_hypercall(first_cpu->env_ptr, hypercall,
                                 sizeof(hypercall));
            _FDT((fdt_property(fdt, "hcall-instructions", hypercall,
                              sizeof(hypercall))));
        }
        _FDT((fdt_end_node(fdt)));
    }
    _FDT((fdt_end_node(fdt))); /* close root node */
    _FDT((fdt_finish(fdt)));
    return fdt;
}