static int nut_probe(AVProbeData *p) {

    if (p->buf_size >= ID_LENGTH && !memcmp(p->buf, ID_STRING, ID_LENGTH)) return AVPROBE_SCORE_MAX;



    return 0;

}
