static int h263_probe(AVProbeData *p)

{

    int code;

    const uint8_t *d;



    if (p->buf_size < 6)

        return 0;

    d = p->buf;

    code = (d[0] << 14) | (d[1] << 6) | (d[2] >> 2);

    if (code == 0x20) {

        return 50;

    }

    return 0;

}
