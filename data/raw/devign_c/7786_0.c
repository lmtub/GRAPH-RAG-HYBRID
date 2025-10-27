void microblaze_load_kernel(MicroBlazeCPU *cpu, target_phys_addr_t ddr_base,

                            uint32_t ramsize, const char *dtb_filename,

                            void (*machine_cpu_reset)(MicroBlazeCPU *))

{

    QemuOpts *machine_opts;

    const char *kernel_filename = NULL;

    const char *kernel_cmdline = NULL;



    machine_opts = qemu_opts_find(qemu_find_opts("machine"), 0);

    if (machine_opts) {

        const char *dtb_arg;

        kernel_filename = qemu_opt_get(machine_opts, "kernel");

        kernel_cmdline = qemu_opt_get(machine_opts, "append");

        dtb_arg = qemu_opt_get(machine_opts, "dtb");

        if (dtb_arg) { /* Preference a -dtb argument */

            dtb_filename = dtb_arg;

        } else { /* default to pcbios dtb as passed by machine_init */

            dtb_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, dtb_filename);

        }

    }



    boot_info.machine_cpu_reset = machine_cpu_reset;

    qemu_register_reset(main_cpu_reset, cpu);



    if (kernel_filename) {

        int kernel_size;

        uint64_t entry, low, high;

        uint32_t base32;

        int big_endian = 0;



#ifdef TARGET_WORDS_BIGENDIAN

        big_endian = 1;

#endif



        /* Boots a kernel elf binary.  */

        kernel_size = load_elf(kernel_filename, NULL, NULL,

                               &entry, &low, &high,

                               big_endian, ELF_MACHINE, 0);

        base32 = entry;

        if (base32 == 0xc0000000) {

            kernel_size = load_elf(kernel_filename, translate_kernel_address,

                                   NULL, &entry, NULL, NULL,

                                   big_endian, ELF_MACHINE, 0);

        }

        /* Always boot into physical ram.  */

        boot_info.bootstrap_pc = ddr_base + (entry & 0x0fffffff);



        /* If it wasn't an ELF image, try an u-boot image.  */

        if (kernel_size < 0) {

            target_phys_addr_t uentry, loadaddr;



            kernel_size = load_uimage(kernel_filename, &uentry, &loadaddr, 0);

            boot_info.bootstrap_pc = uentry;

            high = (loadaddr + kernel_size + 3) & ~3;

        }



        /* Not an ELF image nor an u-boot image, try a RAW image.  */

        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename, ddr_base,

                                              ram_size);

            boot_info.bootstrap_pc = ddr_base;

            high = (ddr_base + kernel_size + 3) & ~3;

        }



        boot_info.cmdline = high + 4096;

        if (kernel_cmdline && strlen(kernel_cmdline)) {

            pstrcpy_targphys("cmdline", boot_info.cmdline, 256, kernel_cmdline);

        }

        /* Provide a device-tree.  */

        boot_info.fdt = boot_info.cmdline + 4096;

        microblaze_load_dtb(boot_info.fdt, ram_size, kernel_cmdline,

                                                     dtb_filename);

    }



}
