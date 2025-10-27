static CharDriverState *qmp_chardev_open_file(ChardevFile *file, Error **errp)

{

    int flags, in = -1, out = -1;



    flags = O_WRONLY | O_TRUNC | O_CREAT | O_BINARY;

    out = qmp_chardev_open_file_source(file->out, flags, errp);

    if (error_is_set(errp)) {

        return NULL;

    }



    if (file->has_in) {

        flags = O_RDONLY;

        in = qmp_chardev_open_file_source(file->in, flags, errp);

        if (error_is_set(errp)) {

            qemu_close(out);

            return NULL;

        }

    }



    return qemu_chr_open_fd(in, out);

}
