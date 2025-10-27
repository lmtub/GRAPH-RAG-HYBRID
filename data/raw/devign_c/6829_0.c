static void ppc_core99_init (ram_addr_t ram_size,

                             const char *boot_device,

                             const char *kernel_filename,

                             const char *kernel_cmdline,

                             const char *initrd_filename,

                             const char *cpu_model)

{

    CPUState *env = NULL, *envs[MAX_CPUS];

    char *filename;

    qemu_irq *pic, **openpic_irqs;

    int unin_memory;

    int linux_boot, i;

    ram_addr_t ram_offset, bios_offset, vga_bios_offset;

    uint32_t kernel_base, kernel_size, initrd_base, initrd_size;

    PCIBus *pci_bus;

    MacIONVRAMState *nvr;

    int nvram_mem_index;

    int vga_bios_size, bios_size;

    int pic_mem_index, dbdma_mem_index, cuda_mem_index, escc_mem_index;

    int ide_mem_index[3];

    int ppc_boot_device;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    void *fw_cfg;

    void *dbdma;

    uint8_t *vga_bios_ptr;

    int machine_arch;



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (cpu_model == NULL)

#ifdef TARGET_PPC64

        cpu_model = "970fx";

#else

        cpu_model = "G4";

#endif

    for (i = 0; i < smp_cpus; i++) {

        env = cpu_init(cpu_model);

        if (!env) {

            fprintf(stderr, "Unable to find PowerPC CPU definition\n");

            exit(1);

        }

        /* Set time-base frequency to 100 Mhz */

        cpu_ppc_tb_init(env, 100UL * 1000UL * 1000UL);

#if 0

        env->osi_call = vga_osi_call;

#endif

        qemu_register_reset((QEMUResetHandler*)&cpu_reset, env);

        envs[i] = env;

    }



    /* allocate RAM */

    ram_offset = qemu_ram_alloc(NULL, "ppc_core99.ram", ram_size);

    cpu_register_physical_memory(0, ram_size, ram_offset);



    /* allocate and load BIOS */

    bios_offset = qemu_ram_alloc(NULL, "ppc_core99.bios", BIOS_SIZE);

    if (bios_name == NULL)

        bios_name = PROM_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    cpu_register_physical_memory(PROM_ADDR, BIOS_SIZE, bios_offset | IO_MEM_ROM);



    /* Load OpenBIOS (ELF) */

    if (filename) {

        bios_size = load_elf(filename, NULL, NULL, NULL,

                             NULL, NULL, 1, ELF_MACHINE, 0);



        qemu_free(filename);

    } else {

        bios_size = -1;

    }

    if (bios_size < 0 || bios_size > BIOS_SIZE) {

        hw_error("qemu: could not load PowerPC bios '%s'\n", bios_name);

        exit(1);

    }



    /* allocate and load VGA BIOS */

    vga_bios_offset = qemu_ram_alloc(NULL, "ppc_core99.vbios", VGA_BIOS_SIZE);

    vga_bios_ptr = qemu_get_ram_ptr(vga_bios_offset);

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, VGABIOS_FILENAME);

    if (filename) {

        vga_bios_size = load_image(filename, vga_bios_ptr + 8);

        qemu_free(filename);

    } else {

        vga_bios_size = -1;

    }

    if (vga_bios_size < 0) {

        /* if no bios is present, we can still work */

        fprintf(stderr, "qemu: warning: could not load VGA bios '%s'\n",

                VGABIOS_FILENAME);

        vga_bios_size = 0;

    } else {

        /* set a specific header (XXX: find real Apple format for NDRV

           drivers) */

        vga_bios_ptr[0] = 'N';

        vga_bios_ptr[1] = 'D';

        vga_bios_ptr[2] = 'R';

        vga_bios_ptr[3] = 'V';

        cpu_to_be32w((uint32_t *)(vga_bios_ptr + 4), vga_bios_size);

        vga_bios_size += 8;



        /* Round to page boundary */

        vga_bios_size = (vga_bios_size + TARGET_PAGE_SIZE - 1) &

            TARGET_PAGE_MASK;

    }



    if (linux_boot) {

        uint64_t lowaddr = 0;

        int bswap_needed;



#ifdef BSWAP_NEEDED

        bswap_needed = 1;

#else

        bswap_needed = 0;

#endif

        kernel_base = KERNEL_LOAD_ADDR;



        kernel_size = load_elf(kernel_filename, translate_kernel_address, NULL,

                               NULL, &lowaddr, NULL, 1, ELF_MACHINE, 0);

        if (kernel_size < 0)

            kernel_size = load_aout(kernel_filename, kernel_base,

                                    ram_size - kernel_base, bswap_needed,

                                    TARGET_PAGE_SIZE);

        if (kernel_size < 0)

            kernel_size = load_image_targphys(kernel_filename,

                                              kernel_base,

                                              ram_size - kernel_base);

        if (kernel_size < 0) {

            hw_error("qemu: could not load kernel '%s'\n", kernel_filename);

            exit(1);

        }

        /* load initrd */

        if (initrd_filename) {

            initrd_base = INITRD_LOAD_ADDR;

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              ram_size - initrd_base);

            if (initrd_size < 0) {

                hw_error("qemu: could not load initial ram disk '%s'\n",

                         initrd_filename);

                exit(1);

            }

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

        ppc_boot_device = 'm';

    } else {

        kernel_base = 0;

        kernel_size = 0;

        initrd_base = 0;

        initrd_size = 0;

        ppc_boot_device = '\0';

        /* We consider that NewWorld PowerMac never have any floppy drive

         * For now, OHW cannot boot from the network.

         */

        for (i = 0; boot_device[i] != '\0'; i++) {

            if (boot_device[i] >= 'c' && boot_device[i] <= 'f') {

                ppc_boot_device = boot_device[i];

                break;

            }

        }

        if (ppc_boot_device == '\0') {

            fprintf(stderr, "No valid boot device for Mac99 machine\n");

            exit(1);

        }

    }



    isa_mem_base = 0x80000000;



    /* Register 8 MB of ISA IO space */

    isa_mmio_init(0xf2000000, 0x00800000, 1);



    /* UniN init */

    unin_memory = cpu_register_io_memory(unin_read, unin_write, NULL);

    cpu_register_physical_memory(0xf8000000, 0x00001000, unin_memory);



    openpic_irqs = qemu_mallocz(smp_cpus * sizeof(qemu_irq *));

    openpic_irqs[0] =

        qemu_mallocz(smp_cpus * sizeof(qemu_irq) * OPENPIC_OUTPUT_NB);

    for (i = 0; i < smp_cpus; i++) {

        /* Mac99 IRQ connection between OpenPIC outputs pins

         * and PowerPC input pins

         */

        switch (PPC_INPUT(env)) {

        case PPC_FLAGS_INPUT_6xx:

            openpic_irqs[i] = openpic_irqs[0] + (i * OPENPIC_OUTPUT_NB);

            openpic_irqs[i][OPENPIC_OUTPUT_INT] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_CINT] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_MCK] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_MCP];

            /* Not connected ? */

            openpic_irqs[i][OPENPIC_OUTPUT_DEBUG] = NULL;

            /* Check this */

            openpic_irqs[i][OPENPIC_OUTPUT_RESET] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_HRESET];

            break;

#if defined(TARGET_PPC64)

        case PPC_FLAGS_INPUT_970:

            openpic_irqs[i] = openpic_irqs[0] + (i * OPENPIC_OUTPUT_NB);

            openpic_irqs[i][OPENPIC_OUTPUT_INT] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_CINT] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_MCK] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_MCP];

            /* Not connected ? */

            openpic_irqs[i][OPENPIC_OUTPUT_DEBUG] = NULL;

            /* Check this */

            openpic_irqs[i][OPENPIC_OUTPUT_RESET] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_HRESET];

            break;

#endif /* defined(TARGET_PPC64) */

        default:

            hw_error("Bus model not supported on mac99 machine\n");

            exit(1);

        }

    }

    pic = openpic_init(NULL, &pic_mem_index, smp_cpus, openpic_irqs, NULL);

    if (PPC_INPUT(env) == PPC_FLAGS_INPUT_970) {

        /* 970 gets a U3 bus */

        pci_bus = pci_pmac_u3_init(pic);

        machine_arch = ARCH_MAC99_U3;

    } else {

        pci_bus = pci_pmac_init(pic);

        machine_arch = ARCH_MAC99;

    }

    /* init basic PC hardware */

    pci_vga_init(pci_bus, vga_bios_offset, vga_bios_size);



    escc_mem_index = escc_init(0x80013000, pic[0x25], pic[0x24],

                               serial_hds[0], serial_hds[1], ESCC_CLOCK, 4);



    for(i = 0; i < nb_nics; i++)

        pci_nic_init_nofail(&nd_table[i], "ne2k_pci", NULL);



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }

    dbdma = DBDMA_init(&dbdma_mem_index);



    /* We only emulate 2 out of 3 IDE controllers for now */

    ide_mem_index[0] = -1;

    hd[0] = drive_get(IF_IDE, 0, 0);

    hd[1] = drive_get(IF_IDE, 0, 1);

    ide_mem_index[1] = pmac_ide_init(hd, pic[0x0d], dbdma, 0x16, pic[0x02]);

    hd[0] = drive_get(IF_IDE, 1, 0);

    hd[1] = drive_get(IF_IDE, 1, 1);

    ide_mem_index[2] = pmac_ide_init(hd, pic[0x0e], dbdma, 0x1a, pic[0x02]);



    /* cuda also initialize ADB */

    if (machine_arch == ARCH_MAC99_U3) {

        usb_enabled = 1;

    }

    cuda_init(&cuda_mem_index, pic[0x19]);



    adb_kbd_init(&adb_bus);

    adb_mouse_init(&adb_bus);



    macio_init(pci_bus, PCI_DEVICE_ID_APPLE_UNI_N_KEYL, 0, pic_mem_index,

               dbdma_mem_index, cuda_mem_index, NULL, 3, ide_mem_index,

               escc_mem_index);



    if (usb_enabled) {

        usb_ohci_init_pci(pci_bus, -1);

    }



    /* U3 needs to use USB for input because Linux doesn't support via-cuda

       on PPC64 */

    if (machine_arch == ARCH_MAC99_U3) {

        usbdevice_create("keyboard");

        usbdevice_create("mouse");

    }



    if (graphic_depth != 15 && graphic_depth != 32 && graphic_depth != 8)

        graphic_depth = 15;



    /* The NewWorld NVRAM is not located in the MacIO device */

    nvr = macio_nvram_init(&nvram_mem_index, 0x2000, 1);

    pmac_format_nvram_partition(nvr, 0x2000);

    macio_nvram_map(nvr, 0xFFF04000);

    /* No PCI init: the BIOS will do it */



    fw_cfg = fw_cfg_init(0, 0, CFG_ADDR, CFG_ADDR + 2);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, machine_arch);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, kernel_base);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_SIZE, kernel_size);

    if (kernel_cmdline) {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, CMDLINE_ADDR);

        pstrcpy_targphys("cmdline", CMDLINE_ADDR, TARGET_PAGE_SIZE, kernel_cmdline);

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, 0);

    }

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, initrd_base);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, initrd_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_BOOT_DEVICE, ppc_boot_device);



    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_WIDTH, graphic_width);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_HEIGHT, graphic_height);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_DEPTH, graphic_depth);



    fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_IS_KVM, kvm_enabled());

    if (kvm_enabled()) {

#ifdef CONFIG_KVM

        uint8_t *hypercall;



        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, kvmppc_get_tbfreq());

        hypercall = qemu_malloc(16);

        kvmppc_get_hypercall(env, hypercall, 16);

        fw_cfg_add_bytes(fw_cfg, FW_CFG_PPC_KVM_HC, hypercall, 16);

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_KVM_PID, getpid());

#endif

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, get_ticks_per_sec());

    }



    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);

}
