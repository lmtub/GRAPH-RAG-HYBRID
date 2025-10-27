static ssize_t drop_sync(int fd, size_t size)

{

    ssize_t ret, dropped = size;

    uint8_t *buffer = g_malloc(MIN(65536, size));



    while (size > 0) {

        ret = read_sync(fd, buffer, MIN(65536, size));

        if (ret < 0) {

            g_free(buffer);

            return ret;

        }



        assert(ret <= size);

        size -= ret;

    }



    g_free(buffer);

    return dropped;

}
