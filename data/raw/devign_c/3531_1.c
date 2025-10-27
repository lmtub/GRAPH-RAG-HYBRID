static QEMUFile *open_test_file(bool write)

{

    int fd = dup(temp_fd);

    QIOChannel *ioc;

    lseek(fd, 0, SEEK_SET);

    if (write) {

        g_assert_cmpint(ftruncate(fd, 0), ==, 0);

    }

    ioc = QIO_CHANNEL(qio_channel_file_new_fd(fd));

    if (write) {

        return qemu_fopen_channel_output(ioc);

    } else {

        return qemu_fopen_channel_input(ioc);

    }

}
