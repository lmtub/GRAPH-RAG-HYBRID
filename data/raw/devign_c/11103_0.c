static int file_write(URLContext *h, const unsigned char *buf, int size)

{

    FileContext *c = h->priv_data;

    int r = write(c->fd, buf, size);

    return (-1 == r)?AVERROR(errno):r;

}
