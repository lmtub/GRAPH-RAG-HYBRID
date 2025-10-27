static int drive_add(const char *file, const char *fmt, ...)

{

    va_list ap;

    int index = drive_opt_get_free_idx();



    if (nb_drives_opt >= MAX_DRIVES || index == -1) {

        fprintf(stderr, "qemu: too many drives\n");

        exit(1);

    }



    drives_opt[index].file = file;

    va_start(ap, fmt);

    vsnprintf(drives_opt[index].opt,

              sizeof(drives_opt[0].opt), fmt, ap);

    va_end(ap);



    nb_drives_opt++;

    return index;

}
