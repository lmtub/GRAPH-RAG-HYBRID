static int set_boot_dev(ISADevice *s, const char *boot_device)

{

#define PC_MAX_BOOT_DEVICES 3

    int nbds, bds[3] = { 0, };

    int i;



    nbds = strlen(boot_device);

    if (nbds > PC_MAX_BOOT_DEVICES) {

        error_report("Too many boot devices for PC");

        return(1);

    }

    for (i = 0; i < nbds; i++) {

        bds[i] = boot_device2nibble(boot_device[i]);

        if (bds[i] == 0) {

            error_report("Invalid boot device for PC: '%c'",

                         boot_device[i]);

            return(1);

        }

    }

    rtc_set_memory(s, 0x3d, (bds[1] << 4) | bds[0]);

    rtc_set_memory(s, 0x38, (bds[2] << 4) | (fd_bootchk ? 0x0 : 0x1));

    return(0);

}
