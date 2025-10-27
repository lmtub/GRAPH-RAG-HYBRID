static int add_old_style_options(const char *fmt, QemuOpts *opts,

                                 const char *base_filename,

                                 const char *base_fmt)

{

    if (base_filename) {

        if (qemu_opt_set(opts, BLOCK_OPT_BACKING_FILE, base_filename)) {

            error_report("Backing file not supported for file format '%s'",

                         fmt);

            return -1;

        }

    }

    if (base_fmt) {

        if (qemu_opt_set(opts, BLOCK_OPT_BACKING_FMT, base_fmt)) {

            error_report("Backing file format not supported for file "

                         "format '%s'", fmt);

            return -1;

        }

    }

    return 0;

}
