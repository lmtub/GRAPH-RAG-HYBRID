static int roq_probe(AVProbeData *p)

{

    if (p->buf_size < 6)

        return 0;



    if ((AV_RL16(&p->buf[0]) != RoQ_MAGIC_NUMBER) ||

        (AV_RL32(&p->buf[2]) != 0xFFFFFFFF))

        return 0;



    return AVPROBE_SCORE_MAX;

}
