static int ape_probe(AVProbeData * p)

{

    if (p->buf[0] == 'M' && p->buf[1] == 'A' && p->buf[2] == 'C' && p->buf[3] == ' ')

        return AVPROBE_SCORE_MAX;



    return 0;

}
