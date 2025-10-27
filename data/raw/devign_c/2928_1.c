static int microdvd_probe(AVProbeData *p)

{

    unsigned char c;

    const uint8_t *ptr = p->buf;

    int i;



    if (AV_RB24(ptr) == 0xEFBBBF)

        ptr += 3;  /* skip UTF-8 BOM */



    for (i=0; i<3; i++) {

        if (sscanf(ptr, "{%*d}{}%c",     &c) != 1 &&

            sscanf(ptr, "{%*d}{%*d}%c",  &c) != 1 &&

            sscanf(ptr, "{DEFAULT}{}%c", &c) != 1)

            return 0;

        ptr += strcspn(ptr, "\n") + 1;

    }

    return AVPROBE_SCORE_MAX;

}
