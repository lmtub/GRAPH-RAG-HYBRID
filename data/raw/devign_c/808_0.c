static int parse_bootdevices(char *devices)

{

    /* We just do some generic consistency checks */

    const char *p;

    int bitmap = 0;



    for (p = devices; *p != '\0'; p++) {

        /* Allowed boot devices are:

         * a-b: floppy disk drives

         * c-f: IDE disk drives

         * g-m: machine implementation dependant drives

         * n-p: network devices

         * It's up to each machine implementation to check if the given boot

         * devices match the actual hardware implementation and firmware

         * features.

         */

        if (*p < 'a' || *p > 'p') {

            fprintf(stderr, "Invalid boot device '%c'\n", *p);

            exit(1);

        }

        if (bitmap & (1 << (*p - 'a'))) {

            fprintf(stderr, "Boot device '%c' was given twice\n", *p);

            exit(1);

        }

        bitmap |= 1 << (*p - 'a');

    }

    return bitmap;

}
