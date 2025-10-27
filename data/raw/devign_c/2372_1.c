static uint8_t get_tlm(Jpeg2000DecoderContext *s, int n)

{

    uint8_t Stlm, ST, SP, tile_tlm, i;

    bytestream_get_byte(&s->buf);               /* Ztlm: skipped */

    Stlm = bytestream_get_byte(&s->buf);



    // too complex ? ST = ((Stlm >> 4) & 0x01) + ((Stlm >> 4) & 0x02);

    ST = (Stlm >> 4) & 0x03;

    // TODO: Manage case of ST = 0b11 --> raise error

    SP       = (Stlm >> 6) & 0x01;

    tile_tlm = (n - 4) / ((SP + 1) * 2 + ST);

    for (i = 0; i < tile_tlm; i++) {

        switch (ST) {

        case 0:

            break;

        case 1:

            bytestream_get_byte(&s->buf);

            break;

        case 2:

            bytestream_get_be16(&s->buf);

            break;

        case 3:

            bytestream_get_be32(&s->buf);

            break;

        }

        if (SP == 0) {

            bytestream_get_be16(&s->buf);

        } else {

            bytestream_get_be32(&s->buf);

        }

    }

    return 0;

}
