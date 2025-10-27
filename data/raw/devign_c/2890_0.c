static int ea_probe(AVProbeData *p)

{

    if (p->buf_size < 4)

        return 0;



    if (AV_RL32(&p->buf[0]) != SCHl_TAG)

        return 0;



    return AVPROBE_SCORE_MAX;

}
