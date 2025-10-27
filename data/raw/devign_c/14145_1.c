static int ast_probe(AVProbeData *p)

{

    if (AV_RL32(p->buf) == MKTAG('S','T','R','M') &&

        AV_RB16(p->buf + 10) &&

        AV_RB16(p->buf + 12) &&

        AV_RB32(p->buf + 16))

        return AVPROBE_SCORE_MAX / 3 * 2;

    return 0;

}
