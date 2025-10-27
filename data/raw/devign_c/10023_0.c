static void mcf5208evb_init(ram_addr_t ram_size, int vga_ram_size,

                     const char *boot_device, DisplayState *ds,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    int kernel_size;

    uint64_t elf_entry;

    target_ulong entry;

    qemu_irq *pic;



    if (!cpu_model)

        cpu_model = "m5208";

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find m68k CPU definition\n");

        exit(1);

    }



    /* Initialize CPU registers.  */

    env->vbr = 0;

    /* TODO: Configure BARs.  */



    /* DRAM at 0x20000000 */

    cpu_register_physical_memory(0x40000000, ram_size,

        qemu_ram_alloc(ram_size) | IO_MEM_RAM);



    /* Internal SRAM.  */

    cpu_register_physical_memory(0x80000000, 16384,

        qemu_ram_alloc(16384) | IO_MEM_RAM);



    /* Internal peripherals.  */

    pic = mcf_intc_init(0xfc048000, env);



    mcf_uart_mm_init(0xfc060000, pic[26], serial_hds[0]);

    mcf_uart_mm_init(0xfc064000, pic[27], serial_hds[1]);

    mcf_uart_mm_init(0xfc068000, pic[28], serial_hds[2]);



    mcf5208_sys_init(pic);



    if (nb_nics > 1) {

        fprintf(stderr, "Too many NICs\n");

        exit(1);

    }

    if (nd_table[0].vlan) {

        if (nd_table[0].model == NULL

            || strcmp(nd_table[0].model, "mcf_fec") == 0) {

            mcf_fec_init(&nd_table[0], 0xfc030000, pic + 36);

        } else if (strcmp(nd_table[0].model, "?") == 0) {

            fprintf(stderr, "qemu: Supported NICs: mcf_fec\n");

            exit (1);

        } else {

            fprintf(stderr, "qemu: Unsupported NIC: %s\n", nd_table[0].model);

            exit (1);

        }

    }



    /*  0xfc000000 SCM.  */

    /*  0xfc004000 XBS.  */

    /*  0xfc008000 FlexBus CS.  */

    /* 0xfc030000 FEC.  */

    /*  0xfc040000 SCM + Power management.  */

    /*  0xfc044000 eDMA.  */

    /* 0xfc048000 INTC.  */

    /*  0xfc058000 I2C.  */

    /*  0xfc05c000 QSPI.  */

    /* 0xfc060000 UART0.  */

    /* 0xfc064000 UART0.  */

    /* 0xfc068000 UART0.  */

    /*  0xfc070000 DMA timers.  */

    /* 0xfc080000 PIT0.  */

    /* 0xfc084000 PIT1.  */

    /*  0xfc088000 EPORT.  */

    /*  0xfc08c000 Watchdog.  */

    /*  0xfc090000 clock module.  */

    /*  0xfc0a0000 CCM + reset.  */

    /*  0xfc0a4000 GPIO.  */

    /* 0xfc0a8000 SDRAM controller.  */



    /* Load kernel.  */

    if (!kernel_filename) {

        fprintf(stderr, "Kernel image must be specified\n");

        exit(1);

    }



    kernel_size = load_elf(kernel_filename, 0, &elf_entry, NULL, NULL);

    entry = elf_entry;

    if (kernel_size < 0) {

        kernel_size = load_uimage(kernel_filename, &entry, NULL, NULL);

    }

    if (kernel_size < 0) {

        kernel_size = load_image(kernel_filename, phys_ram_base);

        entry = 0x20000000;

    }

    if (kernel_size < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n", kernel_filename);

        exit(1);

    }



    env->pc = entry;

}
