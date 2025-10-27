static int64_t load_kernel(void)

{

    int64_t entry, kernel_high;

    long kernel_size, initrd_size, params_size;

    ram_addr_t initrd_offset;

    uint32_t *params_buf;

    int big_endian;



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif

    kernel_size = load_elf(loaderparams.kernel_filename, cpu_mips_kseg0_to_phys,

                           NULL, (uint64_t *)&entry, NULL,

                           (uint64_t *)&kernel_high, big_endian,

                           ELF_MACHINE, 1);

    if (kernel_size >= 0) {

        if ((entry & ~0x7fffffffULL) == 0x80000000)

            entry = (int32_t)entry;

    } else {

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



    /* Store command line.  */

    params_size = 264;

    params_buf = g_malloc(params_size);



    params_buf[0] = tswap32(ram_size);

    params_buf[1] = tswap32(0x12345678);



    if (initrd_size > 0) {

        snprintf((char *)params_buf + 8, 256, "rd_start=0x%" PRIx64 " rd_size=%li %s",

                 cpu_mips_phys_to_kseg0(NULL, initrd_offset),

                 initrd_size, loaderparams.kernel_cmdline);

    } else {

        snprintf((char *)params_buf + 8, 256, "%s", loaderparams.kernel_cmdline);

    }



    rom_add_blob_fixed("params", params_buf, params_size,

                       (16 << 20) - 264);




    return entry;

}