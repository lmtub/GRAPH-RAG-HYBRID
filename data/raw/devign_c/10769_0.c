static int flac_probe(AVProbeData *p)

{

    uint8_t *bufptr = p->buf;



    if(ff_id3v2_match(bufptr))

        bufptr += ff_id3v2_tag_len(bufptr);



    if(memcmp(bufptr, "fLaC", 4)) return 0;

    else                          return AVPROBE_SCORE_MAX / 2;

}
