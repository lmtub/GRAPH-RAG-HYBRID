static int qdraw_probe(AVProbeData *p)

{

    const uint8_t *b = p->buf;



    if (!b[10] && AV_RB32(b+11) == 0x1102ff0c && !b[15] ||

        p->buf_size >= 528 && !b[522] && AV_RB32(b+523) == 0x1102ff0c && !b[527])

        return AVPROBE_SCORE_EXTENSION + 1;

    return 0;

}
