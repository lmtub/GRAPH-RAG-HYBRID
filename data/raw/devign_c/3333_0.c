static void sun4d_hw_init(const struct sun4d_hwdef *hwdef, ram_addr_t RAM_size,

                          const char *boot_device,

                          DisplayState *ds, const char *kernel_filename,

                          const char *kernel_cmdline,

                          const char *initrd_filename, const char *cpu_model)

{

    CPUState *env, *envs[MAX_CPUS];

    unsigned int i;

    void *iounits[MAX_IOUNITS], *espdma, *ledma, *main_esp, *nvram, *sbi;

    qemu_irq *cpu_irqs[MAX_CPUS], *sbi_irq, *sbi_cpu_irq,

        *espdma_irq, *ledma_irq;

    qemu_irq *esp_reset, *le_reset;

    ram_addr_t ram_offset, prom_offset, tcx_offset;

    unsigned long kernel_size;

    int ret;

    char buf[1024];

    int drive_index;

    void *fw_cfg;



    /* init CPUs */

    if (!cpu_model)

        cpu_model = hwdef->default_cpu_model;



    for (i = 0; i < smp_cpus; i++) {

        env = cpu_init(cpu_model);

        if (!env) {

            fprintf(stderr, "qemu: Unable to find Sparc CPU definition\n");

            exit(1);

        }

        cpu_sparc_set_id(env, i);

        envs[i] = env;

        if (i == 0) {

            qemu_register_reset(main_cpu_reset, env);

        } else {

            qemu_register_reset(secondary_cpu_reset, env);

            env->halted = 1;

        }

        cpu_irqs[i] = qemu_allocate_irqs(cpu_set_irq, envs[i], MAX_PILS);

        env->prom_addr = hwdef->slavio_base;

    }



    for (i = smp_cpus; i < MAX_CPUS; i++)

        cpu_irqs[i] = qemu_allocate_irqs(dummy_cpu_set_irq, NULL, MAX_PILS);



    /* allocate RAM */

    if ((uint64_t)RAM_size > hwdef->max_mem) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d, maximum %d\n",

                (unsigned int)(RAM_size / (1024 * 1024)),

                (unsigned int)(hwdef->max_mem / (1024 * 1024)));

        exit(1);

    }

    ram_offset = qemu_ram_alloc(RAM_size);

    cpu_register_physical_memory(0, RAM_size, ram_offset);



    /* load boot prom */

    prom_offset = qemu_ram_alloc(PROM_SIZE_MAX);

    cpu_register_physical_memory(hwdef->slavio_base,

                                 (PROM_SIZE_MAX + TARGET_PAGE_SIZE - 1) &

                                 TARGET_PAGE_MASK,

                                 prom_offset | IO_MEM_ROM);



    if (bios_name == NULL)

        bios_name = PROM_FILENAME;

    snprintf(buf, sizeof(buf), "%s/%s", bios_dir, bios_name);

    ret = load_elf(buf, hwdef->slavio_base - PROM_VADDR, NULL, NULL, NULL);

    if (ret < 0 || ret > PROM_SIZE_MAX)

        ret = load_image_targphys(buf, hwdef->slavio_base, PROM_SIZE_MAX);

    if (ret < 0 || ret > PROM_SIZE_MAX) {

        fprintf(stderr, "qemu: could not load prom '%s'\n",

                buf);

        exit(1);

    }



    /* set up devices */

    sbi = sbi_init(hwdef->sbi_base, &sbi_irq, &sbi_cpu_irq, cpu_irqs);



    for (i = 0; i < MAX_IOUNITS; i++)

        if (hwdef->iounit_bases[i] != (target_phys_addr_t)-1)

            iounits[i] = iommu_init(hwdef->iounit_bases[i],

                                    hwdef->iounit_version,

                                    sbi_irq[hwdef->me_irq]);



    espdma = sparc32_dma_init(hwdef->espdma_base, sbi_irq[hwdef->esp_irq],

                              iounits[0], &espdma_irq, &esp_reset);



    ledma = sparc32_dma_init(hwdef->ledma_base, sbi_irq[hwdef->le_irq],

                             iounits[0], &ledma_irq, &le_reset);



    if (graphic_depth != 8 && graphic_depth != 24) {

        fprintf(stderr, "qemu: Unsupported depth: %d\n", graphic_depth);

        exit (1);

    }

    tcx_offset = qemu_ram_alloc(hwdef->vram_size);

    tcx_init(ds, hwdef->tcx_base, phys_ram_base + tcx_offset, tcx_offset,

             hwdef->vram_size, graphic_width, graphic_height, graphic_depth);



    if (nd_table[0].model == NULL)

        nd_table[0].model = "lance";

    if (strcmp(nd_table[0].model, "lance") == 0) {

        lance_init(&nd_table[0], hwdef->le_base, ledma, *ledma_irq, le_reset);

    } else if (strcmp(nd_table[0].model, "?") == 0) {

        fprintf(stderr, "qemu: Supported NICs: lance\n");

        exit (1);

    } else {

        fprintf(stderr, "qemu: Unsupported NIC: %s\n", nd_table[0].model);

        exit (1);

    }



    nvram = m48t59_init(sbi_irq[0], hwdef->nvram_base, 0,

                        hwdef->nvram_size, 8);



    slavio_timer_init_all(hwdef->counter_base, sbi_irq[hwdef->clock1_irq],

                          sbi_cpu_irq, smp_cpus);



    slavio_serial_ms_kbd_init(hwdef->ms_kb_base, sbi_irq[hwdef->ms_kb_irq],

                              nographic, ESCC_CLOCK, 1);

    // Slavio TTYA (base+4, Linux ttyS0) is the first Qemu serial device

    // Slavio TTYB (base+0, Linux ttyS1) is the second Qemu serial device

    escc_init(hwdef->serial_base, sbi_irq[hwdef->ser_irq], serial_hds[1],

              serial_hds[0], ESCC_CLOCK, 1);



    if (drive_get_max_bus(IF_SCSI) > 0) {

        fprintf(stderr, "qemu: too many SCSI bus\n");

        exit(1);

    }



    main_esp = esp_init(hwdef->esp_base, 2,

                        espdma_memory_read, espdma_memory_write,

                        espdma, *espdma_irq, esp_reset);



    for (i = 0; i < ESP_MAX_DEVS; i++) {

        drive_index = drive_get_index(IF_SCSI, 0, i);

        if (drive_index == -1)

            continue;

        esp_scsi_attach(main_esp, drives_table[drive_index].bdrv, i);

    }



    kernel_size = sun4m_load_kernel(kernel_filename, initrd_filename,

                                    RAM_size);



    nvram_init(nvram, (uint8_t *)&nd_table[0].macaddr, kernel_cmdline,

               boot_device, RAM_size, kernel_size, graphic_width,

               graphic_height, graphic_depth, hwdef->nvram_machine_id,

               "Sun4d");



    fw_cfg = fw_cfg_init(0, 0, CFG_ADDR, CFG_ADDR + 2);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, hwdef->machine_id);

}
