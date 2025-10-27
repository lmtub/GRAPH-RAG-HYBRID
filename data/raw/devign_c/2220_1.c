static int lvf_probe(AVProbeData *p)

{

    if (AV_RL32(p->buf) == MKTAG('L', 'V', 'F', 'F'))

        return AVPROBE_SCORE_EXTENSION;

    return 0;

}
