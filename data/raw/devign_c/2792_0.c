int ff_avc_parse_nal_units(AVIOContext *pb, const uint8_t *buf_in, int size)

{

    const uint8_t *p = buf_in;

    const uint8_t *end = p + size;

    const uint8_t *nal_start, *nal_end;



    size = 0;

    nal_start = ff_avc_find_startcode(p, end);

    while (nal_start < end) {

        while(!*(nal_start++));

        nal_end = ff_avc_find_startcode(nal_start, end);

        avio_wb32(pb, nal_end - nal_start);

        avio_write(pb, nal_start, nal_end - nal_start);

        size += 4 + nal_end - nal_start;

        nal_start = nal_end;

    }

    return size;

}
