static int dxa_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size <= 4)

        return 0;

    if (p->buf[0] == 'D' && p->buf[1] == 'E' &&

        p->buf[2] == 'X' && p->buf[3] == 'A')

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
