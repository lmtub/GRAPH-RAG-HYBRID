static int av_always_inline mlp_thd_probe(AVProbeData *p, uint32_t sync)

{

    const uint8_t *buf, *last_buf = p->buf, *end = p->buf + p->buf_size;

    int frames = 0, valid = 0, size = 0;



    for (buf = p->buf; buf + 8 <= end; buf++) {

        if (AV_RB32(buf + 4) == sync) {

            frames++;

            if (last_buf + size == buf) {

                valid++;

            }

            last_buf = buf;

            size = (AV_RB16(buf) & 0xfff) * 2;

        } else if (buf - last_buf == size) {

            size += (AV_RB16(buf) & 0xfff) * 2;

        }

    }

    if (valid >= 100)

        return AVPROBE_SCORE_MAX;

    return 0;

}
