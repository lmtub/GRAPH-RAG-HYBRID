static void ppc_spapr_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    const char *boot_device = args->boot_device;

    PowerPCCPU *cpu;

    CPUPPCState *env;

    PCIHostState *phb;

    int i;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    hwaddr rma_alloc_size;

    uint32_t initrd_base = 0;

    long kernel_size = 0, initrd_size = 0;

    long load_limit, rtas_limit, fw_size;

    char *filename;



    msi_supported = true;



    spapr = g_malloc0(sizeof(*spapr));

    QLIST_INIT(&spapr->phbs);



    cpu_ppc_hypercall = emulate_spapr_hypercall;



    /* Allocate RMA if necessary */

    rma_alloc_size = kvmppc_alloc_rma("ppc_spapr.rma", sysmem);



    if (rma_alloc_size == -1) {

        hw_error("qemu: Unable to create RMA\n");

        exit(1);

    }



    if (rma_alloc_size && (rma_alloc_size < ram_size)) {

        spapr->rma_size = rma_alloc_size;

    } else {

        spapr->rma_size = ram_size;



        /* With KVM, we don't actually know whether KVM supports an

         * unbounded RMA (PR KVM) or is limited by the hash table size

         * (HV KVM using VRMA), so we always assume the latter

         *

         * In that case, we also limit the initial allocations for RTAS

         * etc... to 256M since we have no way to know what the VRMA size

         * is going to be as it depends on the size of the hash table

         * isn't determined yet.

         */

        if (kvm_enabled()) {

            spapr->vrma_adjust = 1;

            spapr->rma_size = MIN(spapr->rma_size, 0x10000000);

        }

    }



    /* We place the device tree and RTAS just below either the top of the RMA,

     * or just below 2GB, whichever is lowere, so that it can be

     * processed with 32-bit real mode code if necessary */

    rtas_limit = MIN(spapr->rma_size, 0x80000000);

    spapr->rtas_addr = rtas_limit - RTAS_MAX_SIZE;

    spapr->fdt_addr = spapr->rtas_addr - FDT_MAX_SIZE;

    load_limit = spapr->fdt_addr - FW_OVERHEAD;



    /* We aim for a hash table of size 1/128 the size of RAM.  The

     * normal rule of thumb is 1/64 the size of RAM, but that's much

     * more than needed for the Linux guests we support. */

    spapr->htab_shift = 18; /* Minimum architected size */

    while (spapr->htab_shift <= 46) {

        if ((1ULL << (spapr->htab_shift + 7)) >= ram_size) {

            break;

        }

        spapr->htab_shift++;

    }



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = kvm_enabled() ? "host" : "POWER7";

    }

    for (i = 0; i < smp_cpus; i++) {

        cpu = cpu_ppc_init(cpu_model);

        if (cpu == NULL) {

            fprintf(stderr, "Unable to find PowerPC CPU definition\n");

            exit(1);

        }

        env = &cpu->env;



        /* Set time-base frequency to 512 MHz */

        cpu_ppc_tb_init(env, TIMEBASE_FREQ);



        /* PAPR always has exception vectors in RAM not ROM */

        env->hreset_excp_prefix = 0;



        /* Tell KVM that we're in PAPR mode */

        if (kvm_enabled()) {

            kvmppc_set_papr(cpu);

        }



        qemu_register_reset(spapr_cpu_reset, cpu);

    }



    /* allocate RAM */

    spapr->ram_limit = ram_size;

    if (spapr->ram_limit > rma_alloc_size) {

        ram_addr_t nonrma_base = rma_alloc_size;

        ram_addr_t nonrma_size = spapr->ram_limit - rma_alloc_size;



        memory_region_init_ram(ram, "ppc_spapr.ram", nonrma_size);

        vmstate_register_ram_global(ram);

        memory_region_add_subregion(sysmem, nonrma_base, ram);

    }



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, "spapr-rtas.bin");

    spapr->rtas_size = load_image_targphys(filename, spapr->rtas_addr,

                                           rtas_limit - spapr->rtas_addr);

    if (spapr->rtas_size < 0) {

        hw_error("qemu: could not load LPAR rtas '%s'\n", filename);

        exit(1);

    }

    if (spapr->rtas_size > RTAS_MAX_SIZE) {

        hw_error("RTAS too big ! 0x%lx bytes (max is 0x%x)\n",

                 spapr->rtas_size, RTAS_MAX_SIZE);

        exit(1);

    }

    g_free(filename);





    /* Set up Interrupt Controller */

    spapr->icp = xics_system_init(XICS_IRQS);

    spapr->next_irq = XICS_IRQ_BASE;



    /* Set up EPOW events infrastructure */

    spapr_events_init(spapr);



    /* Set up IOMMU */

    spapr_iommu_init();



    /* Set up VIO bus */

    spapr->vio_bus = spapr_vio_bus_init();



    for (i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            spapr_vty_create(spapr->vio_bus, serial_hds[i]);

        }

    }



    /* We always have at least the nvram device on VIO */

    spapr_create_nvram(spapr);



    /* Set up PCI */

    spapr_pci_rtas_init();



    spapr_create_phb(spapr, "pci", SPAPR_PCI_BUID,

                     SPAPR_PCI_MEM_WIN_ADDR,

                     SPAPR_PCI_MEM_WIN_SIZE,

                     SPAPR_PCI_IO_WIN_ADDR,

                     SPAPR_PCI_MSI_WIN_ADDR);

    phb = PCI_HOST_BRIDGE(QLIST_FIRST(&spapr->phbs));



    for (i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!nd->model) {

            nd->model = g_strdup("ibmveth");

        }



        if (strcmp(nd->model, "ibmveth") == 0) {

            spapr_vlan_create(spapr->vio_bus, nd);

        } else {

            pci_nic_init_nofail(&nd_table[i], nd->model, NULL);

        }

    }



    for (i = 0; i <= drive_get_max_bus(IF_SCSI); i++) {

        spapr_vscsi_create(spapr->vio_bus);

    }



    /* Graphics */

    if (spapr_vga_init(phb->bus)) {

        spapr->has_graphics = true;

    }



    if (usb_enabled(spapr->has_graphics)) {

        pci_create_simple(phb->bus, -1, "pci-ohci");

        if (spapr->has_graphics) {

            usbdevice_create("keyboard");

            usbdevice_create("mouse");

        }

    }



    if (spapr->rma_size < (MIN_RMA_SLOF << 20)) {

        fprintf(stderr, "qemu: pSeries SLOF firmware requires >= "

                "%ldM guest RMA (Real Mode Area memory)\n", MIN_RMA_SLOF);

        exit(1);

    }



    if (kernel_filename) {

        uint64_t lowaddr = 0;



        kernel_size = load_elf(kernel_filename, translate_kernel_address, NULL,

                               NULL, &lowaddr, NULL, 1, ELF_MACHINE, 0);

        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename,

                                              KERNEL_LOAD_ADDR,

                                              load_limit - KERNEL_LOAD_ADDR);

        }

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }



        /* load initrd */

        if (initrd_filename) {

            /* Try to locate the initrd in the gap between the kernel

             * and the firmware. Add a bit of space just in case

             */

            initrd_base = (KERNEL_LOAD_ADDR + kernel_size + 0x1ffff) & ~0xffff;

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              load_limit - initrd_base);

            if (initrd_size < 0) {

                fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                        initrd_filename);

                exit(1);

            }

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

    }



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, FW_FILE_NAME);

    fw_size = load_image_targphys(filename, 0, FW_MAX_SIZE);

    if (fw_size < 0) {

        hw_error("qemu: could not load LPAR rtas '%s'\n", filename);

        exit(1);

    }

    g_free(filename);



    spapr->entry_point = 0x100;



    /* Prepare the device tree */

    spapr->fdt_skel = spapr_create_fdt_skel(cpu_model,

                                            initrd_base, initrd_size,

                                            kernel_size,

                                            boot_device, kernel_cmdline,

                                            spapr->epow_irq);

    assert(spapr->fdt_skel != NULL);

}
