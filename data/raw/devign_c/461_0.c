static void sun4uv_init(ram_addr_t RAM_size,

                        const char *boot_devices,

                        const char *kernel_filename, const char *kernel_cmdline,

                        const char *initrd_filename, const char *cpu_model,

                        const struct hwdef *hwdef)

{

    CPUState *env;

    char *filename;

    m48t59_t *nvram;

    int ret, linux_boot;

    unsigned int i;

    ram_addr_t ram_offset, prom_offset;

    long initrd_size, kernel_size;

    PCIBus *pci_bus, *pci_bus2, *pci_bus3;

    QEMUBH *bh;

    qemu_irq *irq;

    int drive_index;

    BlockDriverState *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    BlockDriverState *fd[MAX_FD];

    void *fw_cfg;

    ResetData *reset_info;



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (!cpu_model)

        cpu_model = hwdef->default_cpu_model;



    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find Sparc CPU definition\n");

        exit(1);

    }

    bh = qemu_bh_new(tick_irq, env);

    env->tick = ptimer_init(bh);

    ptimer_set_period(env->tick, 1ULL);



    bh = qemu_bh_new(stick_irq, env);

    env->stick = ptimer_init(bh);

    ptimer_set_period(env->stick, 1ULL);



    bh = qemu_bh_new(hstick_irq, env);

    env->hstick = ptimer_init(bh);

    ptimer_set_period(env->hstick, 1ULL);



    reset_info = qemu_mallocz(sizeof(ResetData));

    reset_info->env = env;

    reset_info->reset_addr = hwdef->prom_addr + 0x40ULL;

    qemu_register_reset(main_cpu_reset, reset_info);

    main_cpu_reset(reset_info);

    // Override warm reset address with cold start address

    env->pc = hwdef->prom_addr + 0x20ULL;

    env->npc = env->pc + 4;



    /* allocate RAM */

    ram_offset = qemu_ram_alloc(RAM_size);

    cpu_register_physical_memory(0, RAM_size, ram_offset);



    prom_offset = qemu_ram_alloc(PROM_SIZE_MAX);

    cpu_register_physical_memory(hwdef->prom_addr,

                                 (PROM_SIZE_MAX + TARGET_PAGE_SIZE) &

                                 TARGET_PAGE_MASK,

                                 prom_offset | IO_MEM_ROM);



    if (bios_name == NULL)

        bios_name = PROM_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        ret = load_elf(filename, hwdef->prom_addr - PROM_VADDR,

                       NULL, NULL, NULL);

        if (ret < 0) {

            ret = load_image_targphys(filename, hwdef->prom_addr,

                                      (PROM_SIZE_MAX + TARGET_PAGE_SIZE) &

                                  TARGET_PAGE_MASK);

        }

        qemu_free(filename);

    } else {

        ret = -1;

    }

    if (ret < 0) {

        fprintf(stderr, "qemu: could not load prom '%s'\n",

                bios_name);

        exit(1);

    }



    kernel_size = 0;

    initrd_size = 0;

    if (linux_boot) {

        /* XXX: put correct offset */

        kernel_size = load_elf(kernel_filename, 0, NULL, NULL, NULL);

        if (kernel_size < 0)

            kernel_size = load_aout(kernel_filename, KERNEL_LOAD_ADDR,

                                    ram_size - KERNEL_LOAD_ADDR);

        if (kernel_size < 0)

            kernel_size = load_image_targphys(kernel_filename,

                                              KERNEL_LOAD_ADDR,

                                              ram_size - KERNEL_LOAD_ADDR);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }



        /* load initrd */

        if (initrd_filename) {

            initrd_size = load_image_targphys(initrd_filename,

                                              INITRD_LOAD_ADDR,

                                              ram_size - INITRD_LOAD_ADDR);

            if (initrd_size < 0) {

                fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                        initrd_filename);

                exit(1);

            }

        }

        if (initrd_size > 0) {

            for (i = 0; i < 64 * TARGET_PAGE_SIZE; i += TARGET_PAGE_SIZE) {

                if (ldl_phys(KERNEL_LOAD_ADDR + i) == 0x48647253) { // HdrS

                    stl_phys(KERNEL_LOAD_ADDR + i + 16, INITRD_LOAD_ADDR);

                    stl_phys(KERNEL_LOAD_ADDR + i + 20, initrd_size);

                    break;

                }

            }

        }

    }

    pci_bus = pci_apb_init(APB_SPECIAL_BASE, APB_MEM_BASE, NULL, &pci_bus2,

                           &pci_bus3);

    isa_mem_base = VGA_BASE;

    pci_vga_init(pci_bus, 0, 0);



    // XXX Should be pci_bus3

    pci_ebus_init(pci_bus, -1);



    i = 0;

    if (hwdef->console_serial_base) {

        serial_mm_init(hwdef->console_serial_base, 0, NULL, 115200,

                       serial_hds[i], 1);

        i++;

    }

    for(; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_init(serial_io[i], NULL/*serial_irq[i]*/, 115200,

                        serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(parallel_io[i], NULL/*parallel_irq[i]*/,

                          parallel_hds[i]);

        }

    }



    for(i = 0; i < nb_nics; i++)

        pci_nic_init(&nd_table[i], "ne2k_pci", NULL);



    irq = qemu_allocate_irqs(cpu_set_irq, env, MAX_PILS);

    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }

    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        drive_index = drive_get_index(IF_IDE, i / MAX_IDE_DEVS,

                                      i % MAX_IDE_DEVS);

       if (drive_index != -1)

           hd[i] = drives_table[drive_index].bdrv;

       else

           hd[i] = NULL;

    }



    pci_cmd646_ide_init(pci_bus, hd, 1);



    /* FIXME: wire up interrupts.  */

    i8042_init(NULL/*1*/, NULL/*12*/, 0x60);

    for(i = 0; i < MAX_FD; i++) {

        drive_index = drive_get_index(IF_FLOPPY, 0, i);

       if (drive_index != -1)

           fd[i] = drives_table[drive_index].bdrv;

       else

           fd[i] = NULL;

    }

    floppy_controller = fdctrl_init(NULL/*6*/, 2, 0, 0x3f0, fd);

    nvram = m48t59_init(NULL/*8*/, 0, 0x0074, NVRAM_SIZE, 59);

    sun4u_NVRAM_set_params(nvram, NVRAM_SIZE, "Sun4u", RAM_size, boot_devices,

                           KERNEL_LOAD_ADDR, kernel_size,

                           kernel_cmdline,

                           INITRD_LOAD_ADDR, initrd_size,

                           /* XXX: need an option to load a NVRAM image */

                           0,

                           graphic_width, graphic_height, graphic_depth,

                           (uint8_t *)&nd_table[0].macaddr);



    fw_cfg = fw_cfg_init(BIOS_CFG_IOPORT, BIOS_CFG_IOPORT + 1, 0, 0);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, hwdef->machine_id);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, KERNEL_LOAD_ADDR);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_SIZE, kernel_size);

    if (kernel_cmdline) {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, CMDLINE_ADDR);

        pstrcpy_targphys(CMDLINE_ADDR, TARGET_PAGE_SIZE, kernel_cmdline);

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, 0);

    }

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, INITRD_LOAD_ADDR);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, initrd_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_BOOT_DEVICE, boot_devices[0]);

    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);

}
