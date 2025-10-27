static const uint8_t *parse_opus_ts_header(const uint8_t *start, int *payload_len, int buf_len)

{

    const uint8_t *buf = start + 1;

    int start_trim_flag, end_trim_flag, control_extension_flag, control_extension_length;

    uint8_t flags;



    GetByteContext gb;

    bytestream2_init(&gb, buf, buf_len);



    flags = bytestream2_get_byte(&gb);

    start_trim_flag        = (flags >> 4) & 1;

    end_trim_flag          = (flags >> 3) & 1;

    control_extension_flag = (flags >> 2) & 1;



    *payload_len = 0;

    while (bytestream2_peek_byte(&gb) == 0xff)

        *payload_len += bytestream2_get_byte(&gb);



    *payload_len += bytestream2_get_byte(&gb);



    if (start_trim_flag)

        bytestream2_skip(&gb, 2);

    if (end_trim_flag)

        bytestream2_skip(&gb, 2);

    if (control_extension_flag) {

        control_extension_length = bytestream2_get_byte(&gb);

        bytestream2_skip(&gb, control_extension_length);

    }



    return buf + bytestream2_tell(&gb);

}
