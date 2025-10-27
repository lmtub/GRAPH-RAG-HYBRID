static int ram_decompress_open(RamDecompressState *s, QEMUFile *f)

{

    int ret;

    memset(s, 0, sizeof(*s));

    s->f = f;

    ret = inflateInit(&s->zstream);

    if (ret != Z_OK)

        return -1;

    return 0;

}
