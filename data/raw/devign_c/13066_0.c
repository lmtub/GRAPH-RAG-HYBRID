static void sun4m_hw_init(const struct sun4m_hwdef *hwdef,

                          MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    unsigned int i;

    void *iommu, *espdma, *ledma, *nvram;

    qemu_irq *cpu_irqs[MAX_CPUS], slavio_irq[32], slavio_cpu_irq[MAX_CPUS],

        espdma_irq, ledma_irq;

    qemu_irq esp_reset, dma_enable;

    qemu_irq fdc_tc;

    qemu_irq *cpu_halt;

    unsigned long kernel_size;

    DriveInfo *fd[MAX_FD];

    FWCfgState *fw_cfg;

    unsigned int num_vsimms;



    /* init CPUs */

    if (!cpu_model)

        cpu_model = hwdef->default_cpu_model;



    for(i = 0; i < smp_cpus; i++) {

        cpu_devinit(cpu_model, i, hwdef->slavio_base, &cpu_irqs[i]);

    }



    for (i = smp_cpus; i < MAX_CPUS; i++)

        cpu_irqs[i] = qemu_allocate_irqs(dummy_cpu_set_irq, NULL, MAX_PILS);





    /* set up devices */

    ram_init(0, machine->ram_size, hwdef->max_mem);

    /* models without ECC don't trap when missing ram is accessed */

    if (!hwdef->ecc_base) {

        empty_slot_init(machine->ram_size, hwdef->max_mem - machine->ram_size);

    }



    prom_init(hwdef->slavio_base, bios_name);



    slavio_intctl = slavio_intctl_init(hwdef->intctl_base,

                                       hwdef->intctl_base + 0x10000ULL,

                                       cpu_irqs);



    for (i = 0; i < 32; i++) {

        slavio_irq[i] = qdev_get_gpio_in(slavio_intctl, i);

    }

    for (i = 0; i < MAX_CPUS; i++) {

        slavio_cpu_irq[i] = qdev_get_gpio_in(slavio_intctl, 32 + i);

    }



    if (hwdef->idreg_base) {

        idreg_init(hwdef->idreg_base);

    }



    if (hwdef->afx_base) {

        afx_init(hwdef->afx_base);

    }



    iommu = iommu_init(hwdef->iommu_base, hwdef->iommu_version,

                       slavio_irq[30]);



    if (hwdef->iommu_pad_base) {

        /* On the real hardware (SS-5, LX) the MMU is not padded, but aliased.

           Software shouldn't use aliased addresses, neither should it crash

           when does. Using empty_slot instead of aliasing can help with

           debugging such accesses */

        empty_slot_init(hwdef->iommu_pad_base,hwdef->iommu_pad_len);

    }



    espdma = sparc32_dma_init(hwdef->dma_base, slavio_irq[18],

                              iommu, &espdma_irq, 0);



    ledma = sparc32_dma_init(hwdef->dma_base + 16ULL,

                             slavio_irq[16], iommu, &ledma_irq, 1);



    if (graphic_depth != 8 && graphic_depth != 24) {

        error_report("Unsupported depth: %d", graphic_depth);

        exit (1);

    }

    num_vsimms = 0;

    if (num_vsimms == 0) {

        if (vga_interface_type == VGA_CG3) {

            if (graphic_depth != 8) {

                error_report("Unsupported depth: %d", graphic_depth);

                exit(1);

            }



            if (!(graphic_width == 1024 && graphic_height == 768) &&

                !(graphic_width == 1152 && graphic_height == 900)) {

                error_report("Unsupported resolution: %d x %d", graphic_width,

                             graphic_height);

                exit(1);

            }



            /* sbus irq 5 */

            cg3_init(hwdef->tcx_base, slavio_irq[11], 0x00100000,

                     graphic_width, graphic_height, graphic_depth);

        } else {

            /* If no display specified, default to TCX */

            if (graphic_depth != 8 && graphic_depth != 24) {

                error_report("Unsupported depth: %d", graphic_depth);

                exit(1);

            }



            if (!(graphic_width == 1024 && graphic_height == 768)) {

                error_report("Unsupported resolution: %d x %d",

                             graphic_width, graphic_height);

                exit(1);

            }



            tcx_init(hwdef->tcx_base, slavio_irq[11], 0x00100000,

                     graphic_width, graphic_height, graphic_depth);

        }

    }



    for (i = num_vsimms; i < MAX_VSIMMS; i++) {

        /* vsimm registers probed by OBP */

        if (hwdef->vsimm[i].reg_base) {

            empty_slot_init(hwdef->vsimm[i].reg_base, 0x2000);

        }

    }



    if (hwdef->sx_base) {

        empty_slot_init(hwdef->sx_base, 0x2000);

    }



    lance_init(&nd_table[0], hwdef->le_base, ledma, ledma_irq);



    nvram = m48t59_init(slavio_irq[0], hwdef->nvram_base, 0, 0x2000, 1968, 8);



    slavio_timer_init_all(hwdef->counter_base, slavio_irq[19], slavio_cpu_irq, smp_cpus);



    slavio_serial_ms_kbd_init(hwdef->ms_kb_base, slavio_irq[14],

                              display_type == DT_NOGRAPHIC, ESCC_CLOCK, 1);

    /* Slavio TTYA (base+4, Linux ttyS0) is the first QEMU serial device

       Slavio TTYB (base+0, Linux ttyS1) is the second QEMU serial device */

    escc_init(hwdef->serial_base, slavio_irq[15], slavio_irq[15],

              serial_hds[0], serial_hds[1], ESCC_CLOCK, 1);



    cpu_halt = qemu_allocate_irqs(cpu_halt_signal, NULL, 1);

    if (hwdef->apc_base) {

        apc_init(hwdef->apc_base, cpu_halt[0]);

    }



    if (hwdef->fd_base) {

        /* there is zero or one floppy drive */

        memset(fd, 0, sizeof(fd));

        fd[0] = drive_get(IF_FLOPPY, 0, 0);

        sun4m_fdctrl_init(slavio_irq[22], hwdef->fd_base, fd,

                          &fdc_tc);

    } else {

        fdc_tc = *qemu_allocate_irqs(dummy_fdc_tc, NULL, 1);

    }



    slavio_misc_init(hwdef->slavio_base, hwdef->aux1_base, hwdef->aux2_base,

                     slavio_irq[30], fdc_tc);



    if (drive_get_max_bus(IF_SCSI) > 0) {

        fprintf(stderr, "qemu: too many SCSI bus\n");

        exit(1);

    }



    esp_init(hwdef->esp_base, 2,

             espdma_memory_read, espdma_memory_write,

             espdma, espdma_irq, &esp_reset, &dma_enable);



    qdev_connect_gpio_out(espdma, 0, esp_reset);

    qdev_connect_gpio_out(espdma, 1, dma_enable);



    if (hwdef->cs_base) {

        sysbus_create_simple("SUNW,CS4231", hwdef->cs_base,

                             slavio_irq[5]);

    }



    if (hwdef->dbri_base) {

        /* ISDN chip with attached CS4215 audio codec */

        /* prom space */

        empty_slot_init(hwdef->dbri_base+0x1000, 0x30);

        /* reg space */

        empty_slot_init(hwdef->dbri_base+0x10000, 0x100);

    }



    if (hwdef->bpp_base) {

        /* parallel port */

        empty_slot_init(hwdef->bpp_base, 0x20);

    }



    kernel_size = sun4m_load_kernel(machine->kernel_filename,

                                    machine->initrd_filename,

                                    machine->ram_size);



    nvram_init(nvram, (uint8_t *)&nd_table[0].macaddr, machine->kernel_cmdline,

               machine->boot_order, machine->ram_size, kernel_size,

               graphic_width, graphic_height, graphic_depth,

               hwdef->nvram_machine_id, "Sun4m");



    if (hwdef->ecc_base)

        ecc_init(hwdef->ecc_base, slavio_irq[28],

                 hwdef->ecc_version);



    fw_cfg = fw_cfg_init_mem(CFG_ADDR, CFG_ADDR + 2);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MAX_CPUS, (uint16_t)max_cpus);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, hwdef->machine_id);

    fw_cfg_add_i16(fw_cfg, FW_CFG_SUN4M_DEPTH, graphic_depth);

    fw_cfg_add_i16(fw_cfg, FW_CFG_SUN4M_WIDTH, graphic_width);

    fw_cfg_add_i16(fw_cfg, FW_CFG_SUN4M_HEIGHT, graphic_height);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, KERNEL_LOAD_ADDR);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_SIZE, kernel_size);

    if (machine->kernel_cmdline) {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, CMDLINE_ADDR);

        pstrcpy_targphys("cmdline", CMDLINE_ADDR, TARGET_PAGE_SIZE,

                         machine->kernel_cmdline);

        fw_cfg_add_string(fw_cfg, FW_CFG_CMDLINE_DATA, machine->kernel_cmdline);

        fw_cfg_add_i32(fw_cfg, FW_CFG_CMDLINE_SIZE,

                       strlen(machine->kernel_cmdline) + 1);

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, 0);

        fw_cfg_add_i32(fw_cfg, FW_CFG_CMDLINE_SIZE, 0);

    }

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, INITRD_LOAD_ADDR);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, 0); // not used

    fw_cfg_add_i16(fw_cfg, FW_CFG_BOOT_DEVICE, machine->boot_order[0]);

    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);

}
