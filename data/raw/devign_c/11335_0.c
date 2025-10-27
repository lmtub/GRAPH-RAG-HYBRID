static int nuv_probe(AVProbeData *p) {

    if (p->buf_size < 12)

        return 0;

    if (!memcmp(p->buf, "NuppelVideo", 12))

        return AVPROBE_SCORE_MAX;

    if (!memcmp(p->buf, "MythTVVideo", 12))

        return AVPROBE_SCORE_MAX;

    return 0;

}
