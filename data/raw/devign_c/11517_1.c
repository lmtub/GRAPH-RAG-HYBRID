static void mpc8544ds_init(ram_addr_t ram_size,

                         const char *boot_device,

                         const char *kernel_filename,

                         const char *kernel_cmdline,

                         const char *initrd_filename,

                         const char *cpu_model)

{

    PCIBus *pci_bus;

    CPUState *env;

    uint64_t elf_entry;

    uint64_t elf_lowaddr;

    target_phys_addr_t entry=0;

    target_phys_addr_t loadaddr=UIMAGE_LOAD_BASE;

    target_long kernel_size=0;

    target_ulong dt_base=DTB_LOAD_BASE;

    target_ulong initrd_base=INITRD_LOAD_BASE;

    target_long initrd_size=0;

    void *fdt;

    int i=0;

    unsigned int pci_irq_nrs[4] = {1, 2, 3, 4};

    qemu_irq *irqs, *mpic, *pci_irqs;

    SerialState * serial[2];



    /* Setup CPU */

    env = cpu_ppc_init("e500v2_v30");

    if (!env) {

        fprintf(stderr, "Unable to initialize CPU!\n");

        exit(1);

    }



    /* Fixup Memory size on a alignment boundary */

    ram_size &= ~(RAM_SIZES_ALIGN - 1);



    /* Register Memory */

    cpu_register_physical_memory(0, ram_size, qemu_ram_alloc(ram_size));



    /* MPIC */

    irqs = qemu_mallocz(sizeof(qemu_irq) * OPENPIC_OUTPUT_NB);

    irqs[OPENPIC_OUTPUT_INT] = ((qemu_irq *)env->irq_inputs)[PPCE500_INPUT_INT];

    irqs[OPENPIC_OUTPUT_CINT] = ((qemu_irq *)env->irq_inputs)[PPCE500_INPUT_CINT];

    mpic = mpic_init(MPC8544_MPIC_REGS_BASE, 1, &irqs, NULL);



    /* Serial */

    if (serial_hds[0])

        serial[0] = serial_mm_init(MPC8544_SERIAL0_REGS_BASE,

                               0, mpic[12+26], 399193,

                        serial_hds[0], 1);



    if (serial_hds[1])

        serial[0] = serial_mm_init(MPC8544_SERIAL1_REGS_BASE,

                        0, mpic[12+26], 399193,

                        serial_hds[0], 1);



    /* PCI */

    pci_irqs = qemu_malloc(sizeof(qemu_irq) * 4);

    pci_irqs[0] = mpic[pci_irq_nrs[0]];

    pci_irqs[1] = mpic[pci_irq_nrs[1]];

    pci_irqs[2] = mpic[pci_irq_nrs[2]];

    pci_irqs[3] = mpic[pci_irq_nrs[3]];

    pci_bus = ppce500_pci_init(pci_irqs, MPC8544_PCI_REGS_BASE);

    if (!pci_bus)

        printf("couldn't create PCI controller!\n");



    isa_mmio_init(MPC8544_PCI_IO, MPC8544_PCI_IOLEN);



    if (pci_bus) {

        /* Register network interfaces. */

        for (i = 0; i < nb_nics; i++) {

            pci_nic_init(&nd_table[i], "virtio", NULL);

        }

    }



    /* Load kernel. */

    if (kernel_filename) {

        kernel_size = load_uimage(kernel_filename, &entry, &loadaddr, NULL);

        if (kernel_size < 0) {

            kernel_size = load_elf(kernel_filename, 0, &elf_entry, &elf_lowaddr,

                                   NULL, 1, ELF_MACHINE, 0);

            entry = elf_entry;

            loadaddr = elf_lowaddr;

        }

        /* XXX try again as binary */

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

    }



    /* Load initrd. */

    if (initrd_filename) {

        initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                          ram_size - initrd_base);



        if (initrd_size < 0) {

            fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                    initrd_filename);

            exit(1);

        }

    }



    /* If we're loading a kernel directly, we must load the device tree too. */

    if (kernel_filename) {

        fdt = mpc8544_load_device_tree(dt_base, ram_size,

                                      initrd_base, initrd_size, kernel_cmdline);

        if (fdt == NULL) {

            fprintf(stderr, "couldn't load device tree\n");

            exit(1);

        }



        /* Set initial guest state. */

        env->gpr[1] = (16<<20) - 8;

        env->gpr[3] = dt_base;

        env->nip = entry;

        /* XXX we currently depend on KVM to create some initial TLB entries. */

    }



    if (kvm_enabled())

        kvmppc_init();



    return;

}
