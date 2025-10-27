static void *spapr_create_fdt_skel(const char *cpu_model,

                                   target_phys_addr_t initrd_base,

                                   target_phys_addr_t initrd_size,

                                   const char *boot_device,

                                   const char *kernel_cmdline,

                                   long hash_shift)

{

    void *fdt;

    CPUState *env;

    uint64_t mem_reg_property[] = { 0, cpu_to_be64(ram_size) };

    uint32_t start_prop = cpu_to_be32(initrd_base);

    uint32_t end_prop = cpu_to_be32(initrd_base + initrd_size);

    uint32_t pft_size_prop[] = {0, cpu_to_be32(hash_shift)};

    char hypertas_prop[] = "hcall-pft\0hcall-term\0hcall-dabr\0hcall-interrupt"

        "\0hcall-tce\0hcall-vio\0hcall-splpar";

    uint32_t interrupt_server_ranges_prop[] = {0, cpu_to_be32(smp_cpus)};

    int i;

    char *modelname;



#define _FDT(exp) \

    do { \

        int ret = (exp);                                           \

        if (ret < 0) {                                             \

            fprintf(stderr, "qemu: error creating device tree: %s: %s\n", \

                    #exp, fdt_strerror(ret));                      \

            exit(1);                                               \

        }                                                          \

    } while (0)



    fdt = g_malloc0(FDT_MAX_SIZE);

    _FDT((fdt_create(fdt, FDT_MAX_SIZE)));



    _FDT((fdt_finish_reservemap(fdt)));



    /* Root node */

    _FDT((fdt_begin_node(fdt, "")));

    _FDT((fdt_property_string(fdt, "device_type", "chrp")));

    _FDT((fdt_property_string(fdt, "model", "IBM pSeries (emulated by qemu)")));



    _FDT((fdt_property_cell(fdt, "#address-cells", 0x2)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x2)));



    /* /chosen */

    _FDT((fdt_begin_node(fdt, "chosen")));



    _FDT((fdt_property_string(fdt, "bootargs", kernel_cmdline)));

    _FDT((fdt_property(fdt, "linux,initrd-start",

                       &start_prop, sizeof(start_prop))));

    _FDT((fdt_property(fdt, "linux,initrd-end",

                       &end_prop, sizeof(end_prop))));

    _FDT((fdt_property_string(fdt, "qemu,boot-device", boot_device)));



    _FDT((fdt_end_node(fdt)));



    /* memory node */

    _FDT((fdt_begin_node(fdt, "memory@0")));



    _FDT((fdt_property_string(fdt, "device_type", "memory")));

    _FDT((fdt_property(fdt, "reg",

                       mem_reg_property, sizeof(mem_reg_property))));



    _FDT((fdt_end_node(fdt)));



    /* cpus */

    _FDT((fdt_begin_node(fdt, "cpus")));



    _FDT((fdt_property_cell(fdt, "#address-cells", 0x1)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x0)));



    modelname = g_strdup(cpu_model);



    for (i = 0; i < strlen(modelname); i++) {

        modelname[i] = toupper(modelname[i]);

    }



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        int index = env->cpu_index;

        uint32_t gserver_prop[] = {cpu_to_be32(index), 0}; /* HACK! */

        char *nodename;

        uint32_t segs[] = {cpu_to_be32(28), cpu_to_be32(40),

                           0xffffffff, 0xffffffff};

        uint32_t tbfreq = kvm_enabled() ? kvmppc_get_tbfreq() : TIMEBASE_FREQ;

        uint32_t cpufreq = kvm_enabled() ? kvmppc_get_clockfreq() : 1000000000;



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

        _FDT((fdt_property(fdt, "ibm,pft-size",

                           pft_size_prop, sizeof(pft_size_prop))));

        _FDT((fdt_property_string(fdt, "status", "okay")));

        _FDT((fdt_property(fdt, "64-bit", NULL, 0)));

        _FDT((fdt_property_cell(fdt, "ibm,ppc-interrupt-server#s", index)));

        _FDT((fdt_property(fdt, "ibm,ppc-interrupt-gserver#s",

                           gserver_prop, sizeof(gserver_prop))));



        if (env->mmu_model & POWERPC_MMU_1TSEG) {

            _FDT((fdt_property(fdt, "ibm,processor-segment-sizes",

                               segs, sizeof(segs))));

        }



        _FDT((fdt_end_node(fdt)));

    }



    g_free(modelname);



    _FDT((fdt_end_node(fdt)));



    /* RTAS */

    _FDT((fdt_begin_node(fdt, "rtas")));



    _FDT((fdt_property(fdt, "ibm,hypertas-functions", hypertas_prop,

                       sizeof(hypertas_prop))));



    _FDT((fdt_end_node(fdt)));



    /* interrupt controller */

    _FDT((fdt_begin_node(fdt, "interrupt-controller@0")));



    _FDT((fdt_property_string(fdt, "device_type",

                              "PowerPC-External-Interrupt-Presentation")));

    _FDT((fdt_property_string(fdt, "compatible", "IBM,ppc-xicp")));

    _FDT((fdt_property_cell(fdt, "reg", 0)));

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



    _FDT((fdt_end_node(fdt))); /* close root node */

    _FDT((fdt_finish(fdt)));



    return fdt;

}
