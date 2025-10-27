static int file_read(URLContext *h, unsigned char *buf, int size)

{

    FileContext *c = h->priv_data;

    int r = read(c->fd, buf, size);

    return (-1 == r)?AVERROR(errno):r;

}
