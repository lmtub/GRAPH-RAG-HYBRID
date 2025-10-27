static void j2k_flush(J2kDecoderContext *s)

{

    if (*s->buf == 0xff)

        s->buf++;

    s->bit_index = 8;

    s->buf++;

}
