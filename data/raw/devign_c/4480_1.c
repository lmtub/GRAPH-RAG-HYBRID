static void sun4uv_init(ram_addr_t RAM_size,

                        const char *boot_devices,

                        const char *kernel_filename, const char *kernel_cmdline,

                        const char *initrd_filename, const char *cpu_model,

                        const struct hwdef *hwdef)

{

    CPUState *env;

    m48t59_t *nvram;

    unsigned int i;

    long initrd_size, kernel_size;

    PCIBus *pci_bus, *pci_bus2, *pci_bus3;

    qemu_irq *irq;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *fd[MAX_FD];

    void *fw_cfg;



    /* init CPUs */

    env = cpu_devinit(cpu_model, hwdef);



    /* set up devices */

    ram_init(0, RAM_size);



    prom_init(hwdef->prom_addr, bios_name);





    irq = qemu_allocate_irqs(cpu_set_irq, env, MAX_PILS);

    pci_bus = pci_apb_init(APB_SPECIAL_BASE, APB_MEM_BASE, irq, &pci_bus2,

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

            serial_isa_init(i, serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(i, parallel_hds[i]);

        }

    }



    for(i = 0; i < nb_nics; i++)

        pci_nic_init(&nd_table[i], "ne2k_pci", NULL);



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }

    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS,

                          i % MAX_IDE_DEVS);

    }



    pci_cmd646_ide_init(pci_bus, hd, 1);



    isa_create_simple("i8042");

    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    fdctrl_init_isa(fd);

    nvram = m48t59_init_isa(0x0074, NVRAM_SIZE, 59);



    initrd_size = 0;

    kernel_size = sun4u_load_kernel(kernel_filename, initrd_filename,

                                    ram_size, &initrd_size);



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



    fw_cfg_add_i16(fw_cfg, FW_CFG_SPARC64_WIDTH, graphic_width);

    fw_cfg_add_i16(fw_cfg, FW_CFG_SPARC64_HEIGHT, graphic_height);

    fw_cfg_add_i16(fw_cfg, FW_CFG_SPARC64_DEPTH, graphic_depth);



    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);

}
