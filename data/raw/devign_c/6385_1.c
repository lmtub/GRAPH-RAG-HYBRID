static void bamboo_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    unsigned int pci_irq_nrs[4] = { 28, 27, 26, 25 };

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *isa = g_new(MemoryRegion, 1);

    MemoryRegion *ram_memories

        = g_malloc(PPC440EP_SDRAM_NR_BANKS * sizeof(*ram_memories));

    hwaddr ram_bases[PPC440EP_SDRAM_NR_BANKS];

    hwaddr ram_sizes[PPC440EP_SDRAM_NR_BANKS];

    qemu_irq *pic;

    qemu_irq *irqs;

    PCIBus *pcibus;

    PowerPCCPU *cpu;

    CPUPPCState *env;

    uint64_t elf_entry;

    uint64_t elf_lowaddr;

    hwaddr loadaddr = 0;

    target_long initrd_size = 0;

    DeviceState *dev;

    int success;

    int i;



    /* Setup CPU. */

    if (machine->cpu_model == NULL) {

        machine->cpu_model = "440EP";

    }

    cpu = POWERPC_CPU(cpu_generic_init(TYPE_POWERPC_CPU, machine->cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "Unable to initialize CPU!\n");

        exit(1);

    }

    env = &cpu->env;



    if (env->mmu_model != POWERPC_MMU_BOOKE) {

        fprintf(stderr, "MMU model %i not supported by this machine.\n",

            env->mmu_model);

        exit(1);

    }



    qemu_register_reset(main_cpu_reset, cpu);

    ppc_booke_timers_init(cpu, 400000000, 0);

    ppc_dcr_init(env, NULL, NULL);



    /* interrupt controller */

    irqs = g_malloc0(sizeof(qemu_irq) * PPCUIC_OUTPUT_NB);

    irqs[PPCUIC_OUTPUT_INT] = ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_INT];

    irqs[PPCUIC_OUTPUT_CINT] = ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_CINT];

    pic = ppcuic_init(env, irqs, 0x0C0, 0, 1);



    /* SDRAM controller */

    memset(ram_bases, 0, sizeof(ram_bases));

    memset(ram_sizes, 0, sizeof(ram_sizes));

    ram_size = ppc4xx_sdram_adjust(ram_size, PPC440EP_SDRAM_NR_BANKS,

                                   ram_memories,

                                   ram_bases, ram_sizes,

                                   ppc440ep_sdram_bank_sizes);

    /* XXX 440EP's ECC interrupts are on UIC1, but we've only created UIC0. */

    ppc4xx_sdram_init(env, pic[14], PPC440EP_SDRAM_NR_BANKS, ram_memories,

                      ram_bases, ram_sizes, 1);



    /* PCI */

    dev = sysbus_create_varargs(TYPE_PPC4xx_PCI_HOST_BRIDGE,

                                PPC440EP_PCI_CONFIG,

                                pic[pci_irq_nrs[0]], pic[pci_irq_nrs[1]],

                                pic[pci_irq_nrs[2]], pic[pci_irq_nrs[3]],

                                NULL);

    pcibus = (PCIBus *)qdev_get_child_bus(dev, "pci.0");

    if (!pcibus) {

        fprintf(stderr, "couldn't create PCI controller!\n");

        exit(1);

    }



    memory_region_init_alias(isa, NULL, "isa_mmio",

                             get_system_io(), 0, PPC440EP_PCI_IOLEN);

    memory_region_add_subregion(get_system_memory(), PPC440EP_PCI_IO, isa);



    if (serial_hds[0] != NULL) {

        serial_mm_init(address_space_mem, 0xef600300, 0, pic[0],

                       PPC_SERIAL_MM_BAUDBASE, serial_hds[0],

                       DEVICE_BIG_ENDIAN);

    }

    if (serial_hds[1] != NULL) {

        serial_mm_init(address_space_mem, 0xef600400, 0, pic[1],

                       PPC_SERIAL_MM_BAUDBASE, serial_hds[1],

                       DEVICE_BIG_ENDIAN);

    }



    if (pcibus) {

        /* Register network interfaces. */

        for (i = 0; i < nb_nics; i++) {

            /* There are no PCI NICs on the Bamboo board, but there are

             * PCI slots, so we can pick whatever default model we want. */

            pci_nic_init_nofail(&nd_table[i], pcibus, "e1000", NULL);

        }

    }



    /* Load kernel. */

    if (kernel_filename) {

        success = load_uimage(kernel_filename, &entry, &loadaddr, NULL,

                              NULL, NULL);

        if (success < 0) {

            success = load_elf(kernel_filename, NULL, NULL, &elf_entry,

                               &elf_lowaddr, NULL, 1, PPC_ELF_MACHINE,

                               0, 0);

            entry = elf_entry;

            loadaddr = elf_lowaddr;

        }

        /* XXX try again as binary */

        if (success < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

    }



    /* Load initrd. */

    if (initrd_filename) {

        initrd_size = load_image_targphys(initrd_filename, RAMDISK_ADDR,

                                          ram_size - RAMDISK_ADDR);



        if (initrd_size < 0) {

            fprintf(stderr, "qemu: could not load ram disk '%s' at %x\n",

                    initrd_filename, RAMDISK_ADDR);

            exit(1);

        }

    }



    /* If we're loading a kernel directly, we must load the device tree too. */

    if (kernel_filename) {

        if (bamboo_load_device_tree(FDT_ADDR, ram_size, RAMDISK_ADDR,

                                    initrd_size, kernel_cmdline) < 0) {

            fprintf(stderr, "couldn't load device tree\n");

            exit(1);

        }

    }

}
