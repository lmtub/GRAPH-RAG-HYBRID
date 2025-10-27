static void set_kernel_args(const struct arm_boot_info *info)

{

    int initrd_size = info->initrd_size;

    target_phys_addr_t base = info->loader_start;

    target_phys_addr_t p;



    p = base + KERNEL_ARGS_ADDR;

    /* ATAG_CORE */

    WRITE_WORD(p, 5);

    WRITE_WORD(p, 0x54410001);

    WRITE_WORD(p, 1);

    WRITE_WORD(p, 0x1000);

    WRITE_WORD(p, 0);

    /* ATAG_MEM */

    /* TODO: handle multiple chips on one ATAG list */

    WRITE_WORD(p, 4);

    WRITE_WORD(p, 0x54410002);

    WRITE_WORD(p, info->ram_size);

    WRITE_WORD(p, info->loader_start);

    if (initrd_size) {

        /* ATAG_INITRD2 */

        WRITE_WORD(p, 4);

        WRITE_WORD(p, 0x54420005);

        WRITE_WORD(p, info->loader_start + INITRD_LOAD_ADDR);

        WRITE_WORD(p, initrd_size);

    }

    if (info->kernel_cmdline && *info->kernel_cmdline) {

        /* ATAG_CMDLINE */

        int cmdline_size;



        cmdline_size = strlen(info->kernel_cmdline);

        cpu_physical_memory_write(p + 8, (void *)info->kernel_cmdline,

                                  cmdline_size + 1);

        cmdline_size = (cmdline_size >> 2) + 1;

        WRITE_WORD(p, cmdline_size + 2);

        WRITE_WORD(p, 0x54410009);

        p += cmdline_size * 4;

    }

    if (info->atag_board) {

        /* ATAG_BOARD */

        int atag_board_len;

        uint8_t atag_board_buf[0x1000];



        atag_board_len = (info->atag_board(info, atag_board_buf) + 3) & ~3;

        WRITE_WORD(p, (atag_board_len + 8) >> 2);

        WRITE_WORD(p, 0x414f4d50);

        cpu_physical_memory_write(p, atag_board_buf, atag_board_len);

        p += atag_board_len;

    }

    /* ATAG_END */

    WRITE_WORD(p, 0);

    WRITE_WORD(p, 0);

}
