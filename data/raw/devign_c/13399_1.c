static int smacker_probe(AVProbeData *p)

{

    if(p->buf[0] == 'S' && p->buf[1] == 'M' && p->buf[2] == 'K'

        && (p->buf[3] == '2' || p->buf[3] == '4'))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
