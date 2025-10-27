static int film_probe(AVProbeData *p)

{

    if (p->buf_size < 4)

        return 0;



    if (AV_RB32(&p->buf[0]) != FILM_TAG)

        return 0;



    return AVPROBE_SCORE_MAX;

}
