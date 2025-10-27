static void write_strip_header(CinepakEncContext *s, int y, int h, int keyframe, unsigned char *buf, int strip_size)

{

    buf[0] = keyframe ? 0x11: 0x10;

    AV_WB24(&buf[1], strip_size + STRIP_HEADER_SIZE);

    AV_WB16(&buf[4], y);

    AV_WB16(&buf[6], 0);

    AV_WB16(&buf[8], h);

    AV_WB16(&buf[10], s->w);

}
