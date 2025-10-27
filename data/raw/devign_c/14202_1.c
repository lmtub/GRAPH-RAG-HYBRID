static int avr_probe(AVProbeData *p)

{

    if (AV_RL32(p->buf) == MKTAG('2', 'B', 'I', 'T'))

        return AVPROBE_SCORE_EXTENSION;

    return 0;

}
