static int64_t load_kernel (void)

{

    int64_t kernel_entry, kernel_high;

    long initrd_size;

    ram_addr_t initrd_offset;

    int big_endian;

    uint32_t *prom_buf;

    long prom_size;

    int prom_index = 0;



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif



    if (load_elf(loaderparams.kernel_filename, cpu_mips_kseg0_to_phys, NULL,

                 (uint64_t *)&kernel_entry, NULL, (uint64_t *)&kernel_high,

                 big_endian, ELF_MACHINE, 1) < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n",

                loaderparams.kernel_filename);

        exit(1);

    }



    /* load initrd */

    initrd_size = 0;

    initrd_offset = 0;

    if (loaderparams.initrd_filename) {

        initrd_size = get_image_size (loaderparams.initrd_filename);

        if (initrd_size > 0) {

            initrd_offset = (kernel_high + ~INITRD_PAGE_MASK) & INITRD_PAGE_MASK;

            if (initrd_offset + initrd_size > ram_size) {

                fprintf(stderr,

                        "qemu: memory too small for initial ram disk '%s'\n",

                        loaderparams.initrd_filename);

                exit(1);

            }

            initrd_size = load_image_targphys(loaderparams.initrd_filename,

                                              initrd_offset,

                                              ram_size - initrd_offset);

        }

        if (initrd_size == (target_ulong) -1) {

            fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                    loaderparams.initrd_filename);

            exit(1);

        }

    }



    /* Setup prom parameters. */

    prom_size = ENVP_NB_ENTRIES * (sizeof(int32_t) + ENVP_ENTRY_SIZE);

    prom_buf = g_malloc(prom_size);



    prom_set(prom_buf, prom_index++, "%s", loaderparams.kernel_filename);

    if (initrd_size > 0) {

        prom_set(prom_buf, prom_index++, "rd_start=0x%" PRIx64 " rd_size=%li %s",

                 cpu_mips_phys_to_kseg0(NULL, initrd_offset), initrd_size,

                 loaderparams.kernel_cmdline);

    } else {

        prom_set(prom_buf, prom_index++, "%s", loaderparams.kernel_cmdline);

    }



    prom_set(prom_buf, prom_index++, "memsize");

    prom_set(prom_buf, prom_index++, "%i", loaderparams.ram_size);

    prom_set(prom_buf, prom_index++, "modetty0");

    prom_set(prom_buf, prom_index++, "38400n8r");

    prom_set(prom_buf, prom_index++, NULL);



    rom_add_blob_fixed("prom", prom_buf, prom_size,

                       cpu_mips_kseg0_to_phys(NULL, ENVP_ADDR));



    return kernel_entry;

}
