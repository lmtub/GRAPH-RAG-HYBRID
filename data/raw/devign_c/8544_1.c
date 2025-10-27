static int ac3_probe(AVProbeData *p)

{

    int max_frames, first_frames, frames;

    uint8_t *buf, *buf2, *end;

    AC3HeaderInfo hdr;



    if(p->buf_size < 7)

        return 0;



    max_frames = 0;

    buf = p->buf;

    end = buf + FFMIN(4096, p->buf_size - 7);



    for(; buf < end; buf++) {

        buf2 = buf;



        for(frames = 0; buf2 < end; frames++) {

            if(ff_ac3_parse_header(buf2, &hdr) < 0)

                break;

            buf2 += hdr.frame_size;

        }

        max_frames = FFMAX(max_frames, frames);

        if(buf == p->buf)

            first_frames = frames;

    }

    if   (first_frames>=3) return AVPROBE_SCORE_MAX * 3 / 4;

    else if(max_frames>=3) return AVPROBE_SCORE_MAX / 2;

    else if(max_frames>=1) return 1;

    else                   return 0;

}
