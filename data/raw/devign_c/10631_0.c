int avio_put_str16le(AVIOContext *s, const char *str)

{

    const uint8_t *q = str;

    int ret = 0;



    while (*q) {

        uint32_t ch;

        uint16_t tmp;



        GET_UTF8(ch, *q++, break;)

        PUT_UTF16(ch, tmp, avio_wl16(s, tmp); ret += 2;)

    }

    avio_wl16(s, 0);

    ret += 2;

    return ret;

}
