static int probe(AVProbeData *p)

{

    if (p->buf_size < 13)

        return 0;



    if (p->buf[0] == 0x01 && p->buf[1] == 0x00 &&

        p->buf[4] == 0x01 + p->buf[2] &&

        p->buf[8] == p->buf[4] + p->buf[6] &&

        p->buf[12] == p->buf[8] + p->buf[10])

        return AVPROBE_SCORE_MAX;



    return 0;

}
