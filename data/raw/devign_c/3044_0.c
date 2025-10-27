void pc_cmos_init(ram_addr_t ram_size, ram_addr_t above_4g_mem_size,

                  const char *boot_device, MachineState *machine,

                  ISADevice *floppy, BusState *idebus0, BusState *idebus1,

                  ISADevice *s)

{

    int val, nb, i;

    FDriveType fd_type[2] = { FDRIVE_DRV_NONE, FDRIVE_DRV_NONE };

    static pc_cmos_init_late_arg arg;

    PCMachineState *pc_machine = PC_MACHINE(machine);



    /* various important CMOS locations needed by PC/Bochs bios */



    /* memory size */

    /* base memory (first MiB) */

    val = MIN(ram_size / 1024, 640);

    rtc_set_memory(s, 0x15, val);

    rtc_set_memory(s, 0x16, val >> 8);

    /* extended memory (next 64MiB) */

    if (ram_size > 1024 * 1024) {

        val = (ram_size - 1024 * 1024) / 1024;

    } else {

        val = 0;

    }

    if (val > 65535)

        val = 65535;

    rtc_set_memory(s, 0x17, val);

    rtc_set_memory(s, 0x18, val >> 8);

    rtc_set_memory(s, 0x30, val);

    rtc_set_memory(s, 0x31, val >> 8);

    /* memory between 16MiB and 4GiB */

    if (ram_size > 16 * 1024 * 1024) {

        val = (ram_size - 16 * 1024 * 1024) / 65536;

    } else {

        val = 0;

    }

    if (val > 65535)

        val = 65535;

    rtc_set_memory(s, 0x34, val);

    rtc_set_memory(s, 0x35, val >> 8);

    /* memory above 4GiB */

    val = above_4g_mem_size / 65536;

    rtc_set_memory(s, 0x5b, val);

    rtc_set_memory(s, 0x5c, val >> 8);

    rtc_set_memory(s, 0x5d, val >> 16);



    /* set the number of CPU */

    rtc_set_memory(s, 0x5f, smp_cpus - 1);



    object_property_add_link(OBJECT(machine), "rtc_state",

                             TYPE_ISA_DEVICE,

                             (Object **)&pc_machine->rtc,

                             object_property_allow_set_link,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE, &error_abort);

    object_property_set_link(OBJECT(machine), OBJECT(s),

                             "rtc_state", &error_abort);



    if (set_boot_dev(s, boot_device)) {

        exit(1);

    }



    /* floppy type */

    if (floppy) {

        for (i = 0; i < 2; i++) {

            fd_type[i] = isa_fdc_get_drive_type(floppy, i);

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

    arg.idebus[0] = idebus0;

    arg.idebus[1] = idebus1;

    qemu_register_reset(pc_cmos_init_late, &arg);

}
