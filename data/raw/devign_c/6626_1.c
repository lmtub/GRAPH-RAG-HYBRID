static int film_probe(AVProbeData *p)
{
    if (AV_RB32(&p->buf[0]) != FILM_TAG)
    return AVPROBE_SCORE_MAX;
}