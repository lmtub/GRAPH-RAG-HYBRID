static int qemu_chr_open_file_out(QemuOpts *opts, CharDriverState **_chr)

{

    int fd_out;



    TFR(fd_out = qemu_open(qemu_opt_get(opts, "path"),

                      O_WRONLY | O_TRUNC | O_CREAT | O_BINARY, 0666));

    if (fd_out < 0) {

        return -errno;

    }



    *_chr = qemu_chr_open_fd(-1, fd_out);

    return 0;

}
