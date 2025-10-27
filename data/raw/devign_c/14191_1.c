static void bamboo_init(ram_addr_t ram_size,

                        const char *boot_device,

                        const char *kernel_filename,

                        const char *kernel_cmdline,

                        const char *initrd_filename,

                        const char *cpu_model)

{

    unsigned int pci_irq_nrs[4] = { 28, 27, 26, 25 };

    PCIBus *pcibus;

    CPUState *env;

    uint64_t elf_entry;

    uint64_t elf_lowaddr;

    target_phys_addr_t entry = 0;

    target_phys_addr_t loadaddr = 0;

    target_long initrd_size = 0;

    int success;

    int i;



    /* Setup CPU. */

    env = ppc440ep_init(&ram_size, &pcibus, pci_irq_nrs, 1, cpu_model);



    if (pcibus) {

        /* Register network interfaces. */

        for (i = 0; i < nb_nics; i++) {

            /* There are no PCI NICs on the Bamboo board, but there are

             * PCI slots, so we can pick whatever default model we want. */

            pci_nic_init_nofail(&nd_table[i], "e1000", NULL);

        }

    }



    /* Load kernel. */

    if (kernel_filename) {

        success = load_uimage(kernel_filename, &entry, &loadaddr, NULL);

        if (success < 0) {

            success = load_elf(kernel_filename, NULL, NULL, &elf_entry,

                               &elf_lowaddr, NULL, 1, ELF_MACHINE, 0);

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



        cpu_synchronize_state(env);



        /* Set initial guest state. */

        env->gpr[1] = (16<<20) - 8;

        env->gpr[3] = FDT_ADDR;

        env->nip = entry;

        /* XXX we currently depend on KVM to create some initial TLB entries. */

    }



    if (kvm_enabled())

        kvmppc_init();

}
