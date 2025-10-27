static int jacosub_probe(AVProbeData *p)

{

    const char *ptr     = p->buf;

    const char *ptr_end = p->buf + p->buf_size;



    if (AV_RB24(ptr) == 0xEFBBBF)

        ptr += 3; /* skip UTF-8 BOM */



    while (ptr < ptr_end) {

        while (jss_whitespace(*ptr))

            ptr++;

        if (*ptr != '#' && *ptr != '\n') {

            if (timed_line(ptr))

                return AVPROBE_SCORE_EXTENSION + 1;

            return 0;

        }

        ptr += strcspn(ptr, "\n") + 1;

    }

    return 0;

}
