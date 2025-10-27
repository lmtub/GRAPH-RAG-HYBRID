void pc_cmos_init(ram_addr_t ram_size, ram_addr_t above_4g_mem_size,

                  const char *boot_device,

                  BusState *idebus0, BusState *idebus1,

                  ISADevice *s)

{

    int val, nb, nb_heads, max_track, last_sect, i;

    FDriveType fd_type[2];

    DriveInfo *fd[2];

    static pc_cmos_init_late_arg arg;



    /* various important CMOS locations needed by PC/Bochs bios */



    /* memory size */

    val = 640; /* base memory in K */

    rtc_set_memory(s, 0x15, val);

    rtc_set_memory(s, 0x16, val >> 8);



    val = (ram_size / 1024) - 1024;

    if (val > 65535)

        val = 65535;

    rtc_set_memory(s, 0x17, val);

    rtc_set_memory(s, 0x18, val >> 8);

    rtc_set_memory(s, 0x30, val);

    rtc_set_memory(s, 0x31, val >> 8);



    if (above_4g_mem_size) {

        rtc_set_memory(s, 0x5b, (unsigned int)above_4g_mem_size >> 16);

        rtc_set_memory(s, 0x5c, (unsigned int)above_4g_mem_size >> 24);

        rtc_set_memory(s, 0x5d, (uint64_t)above_4g_mem_size >> 32);

    }



    if (ram_size > (16 * 1024 * 1024))

        val = (ram_size / 65536) - ((16 * 1024 * 1024) / 65536);

    else

        val = 0;

    if (val > 65535)

        val = 65535;

    rtc_set_memory(s, 0x34, val);

    rtc_set_memory(s, 0x35, val >> 8);



    /* set the number of CPU */

    rtc_set_memory(s, 0x5f, smp_cpus - 1);



    /* set boot devices, and disable floppy signature check if requested */

    if (set_boot_dev(s, boot_device, fd_bootchk)) {

        exit(1);

    }



    /* floppy type */

    for (i = 0; i < 2; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

        if (fd[i]) {

            bdrv_get_floppy_geometry_hint(fd[i]->bdrv, &nb_heads, &max_track,

                                          &last_sect, FDRIVE_DRV_NONE,

                                          &fd_type[i]);

        } else {

            fd_type[i] = FDRIVE_DRV_NONE;

        }

    }

    val = (cmos_get_fd_drive_type(fd_type[0]) << 4) |

        cmos_get_fd_drive_type(fd_type[1]);

    rtc_set_memory(s, 0x10, val);



    val = 0;

    nb = 0;

    if (fd_type[0] < FDRIVE_DRV_NONE) {

        nb++;

    }

    if (fd_type[1] < FDRIVE_DRV_NONE) {

        nb++;

    }

    switch (nb) {

    case 0:

        break;

    case 1:

        val |= 0x01; /* 1 drive, ready for boot */

        break;

    case 2:

        val |= 0x41; /* 2 drives, ready for boot */

        break;

    }

    val |= 0x02; /* FPU is there */

    val |= 0x04; /* PS/2 mouse installed */

    rtc_set_memory(s, REG_EQUIPMENT_BYTE, val);



    /* hard drives */

    arg.rtc_state = s;

    arg.idebus0 = idebus0;

    arg.idebus1 = idebus1;

    qemu_register_reset(pc_cmos_init_late, &arg);

}
