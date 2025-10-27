static int qemu_chr_open_pipe(QemuOpts *opts, CharDriverState **_chr)

{

    int fd_in, fd_out;

    char filename_in[256], filename_out[256];

    const char *filename = qemu_opt_get(opts, "path");



    if (filename == NULL) {

        fprintf(stderr, "chardev: pipe: no filename given\n");

        return -EINVAL;

    }



    snprintf(filename_in, 256, "%s.in", filename);

    snprintf(filename_out, 256, "%s.out", filename);

    TFR(fd_in = qemu_open(filename_in, O_RDWR | O_BINARY));

    TFR(fd_out = qemu_open(filename_out, O_RDWR | O_BINARY));

    if (fd_in < 0 || fd_out < 0) {

	if (fd_in >= 0)

	    close(fd_in);

	if (fd_out >= 0)

	    close(fd_out);

        TFR(fd_in = fd_out = qemu_open(filename, O_RDWR | O_BINARY));

        if (fd_in < 0) {

            return -errno;

        }

    }



    *_chr = qemu_chr_open_fd(fd_in, fd_out);

    return 0;

}
