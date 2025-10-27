static int mpl2_probe(AVProbeData *p)

{

    int i;

    char c;

    int64_t start, end;

    const unsigned char *ptr = p->buf;

    const unsigned char *ptr_end = ptr + p->buf_size;



    for (i = 0; i < 2; i++) {

        if (sscanf(ptr, "[%"SCNd64"][%"SCNd64"]%c", &start, &end, &c) != 3 &&

            sscanf(ptr, "[%"SCNd64"][]%c",          &start,       &c) != 2)

            return 0;

        ptr += strcspn(ptr, "\n") + 1;

        if (ptr >= ptr_end)

            return 0;

    }

    return AVPROBE_SCORE_MAX;

}
