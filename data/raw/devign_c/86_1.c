static int raw_create(const char *filename, QEMUOptionParameter *options)

{

    int fd;

    int64_t total_size = 0;



    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            total_size = options->value.n / 512;

        }

        options++;

    }



    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY,

              0644);

    if (fd < 0)

        return -EIO;

    ftruncate(fd, total_size * 512);

    close(fd);

    return 0;

}
