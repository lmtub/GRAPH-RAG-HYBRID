int boot_sector_init(char *fname)

{

    int fd, ret;

    size_t len = sizeof boot_sector;



    fd = mkstemp(fname);

    if (fd < 0) {

        fprintf(stderr, "Couldn't open \"%s\": %s", fname, strerror(errno));

        return 1;

    }



    /* For Open Firmware based system, we can use a Forth script instead */

    if (strcmp(qtest_get_arch(), "ppc64") == 0) {

        len = sprintf((char *)boot_sector, "\\ Bootscript\n%x %x c! %x %x c!\n",

                LOW(SIGNATURE), BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET,

                HIGH(SIGNATURE), BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET + 1);

    }



    ret = write(fd, boot_sector, len);

    close(fd);



    if (ret != len) {

        fprintf(stderr, "Could not write \"%s\"", fname);

        return 1;

    }



    return 0;

}
