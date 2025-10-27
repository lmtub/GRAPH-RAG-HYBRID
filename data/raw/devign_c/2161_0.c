static void set_kernel_args_old(const struct arm_boot_info *info)

{

    target_phys_addr_t p;

    const char *s;

    int initrd_size = info->initrd_size;

    target_phys_addr_t base = info->loader_start;



    /* see linux/include/asm-arm/setup.h */

    p = base + KERNEL_ARGS_ADDR;

    /* page_size */

    WRITE_WORD(p, 4096);

    /* nr_pages */

    WRITE_WORD(p, info->ram_size / 4096);

    /* ramdisk_size */

    WRITE_WORD(p, 0);

#define FLAG_READONLY	1

#define FLAG_RDLOAD	4

#define FLAG_RDPROMPT	8

    /* flags */

    WRITE_WORD(p, FLAG_READONLY | FLAG_RDLOAD | FLAG_RDPROMPT);

    /* rootdev */

    WRITE_WORD(p, (31 << 8) | 0);	/* /dev/mtdblock0 */

    /* video_num_cols */

    WRITE_WORD(p, 0);

    /* video_num_rows */

    WRITE_WORD(p, 0);

    /* video_x */

    WRITE_WORD(p, 0);

    /* video_y */

    WRITE_WORD(p, 0);

    /* memc_control_reg */

    WRITE_WORD(p, 0);

    /* unsigned char sounddefault */

    /* unsigned char adfsdrives */

    /* unsigned char bytes_per_char_h */

    /* unsigned char bytes_per_char_v */

    WRITE_WORD(p, 0);

    /* pages_in_bank[4] */

    WRITE_WORD(p, 0);

    WRITE_WORD(p, 0);

    WRITE_WORD(p, 0);

    WRITE_WORD(p, 0);

    /* pages_in_vram */

    WRITE_WORD(p, 0);

    /* initrd_start */

    if (initrd_size)

        WRITE_WORD(p, info->loader_start + INITRD_LOAD_ADDR);

    else

        WRITE_WORD(p, 0);

    /* initrd_size */

    WRITE_WORD(p, initrd_size);

    /* rd_start */

    WRITE_WORD(p, 0);

    /* system_rev */

    WRITE_WORD(p, 0);

    /* system_serial_low */

    WRITE_WORD(p, 0);

    /* system_serial_high */

    WRITE_WORD(p, 0);

    /* mem_fclk_21285 */

    WRITE_WORD(p, 0);

    /* zero unused fields */

    while (p < base + KERNEL_ARGS_ADDR + 256 + 1024) {

        WRITE_WORD(p, 0);

    }

    s = info->kernel_cmdline;

    if (s) {

        cpu_physical_memory_write(p, (void *)s, strlen(s) + 1);

    } else {

        WRITE_WORD(p, 0);

    }

}
