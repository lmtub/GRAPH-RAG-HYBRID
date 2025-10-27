static int flv_probe(AVProbeData *p)

{

    const uint8_t *d;



    if (p->buf_size < 6)

        return 0;

    d = p->buf;

    if (d[0] == 'F' && d[1] == 'L' && d[2] == 'V' && d[3] < 5 && d[5]==0) {

        return AVPROBE_SCORE_MAX;

    }

    return 0;

}
