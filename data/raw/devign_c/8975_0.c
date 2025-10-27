static int get_sot(Jpeg2000DecoderContext *s, int n)

{

    Jpeg2000TilePart *tp;

    uint16_t Isot;

    uint32_t Psot;

    uint8_t TPsot;



    if (bytestream2_get_bytes_left(&s->g) < 8)

        return AVERROR_INVALIDDATA;



    s->curtileno = 0;

    Isot = bytestream2_get_be16u(&s->g);        // Isot

    if (Isot >= s->numXtiles * s->numYtiles)

        return AVERROR_INVALIDDATA;



    s->curtileno = Isot;

    Psot  = bytestream2_get_be32u(&s->g);       // Psot

    TPsot = bytestream2_get_byteu(&s->g);       // TPsot



    /* Read TNSot but not used */

    bytestream2_get_byteu(&s->g);               // TNsot



    if (!Psot)

        Psot = bytestream2_get_bytes_left(&s->g) + n + 2;



    if (Psot > bytestream2_get_bytes_left(&s->g) + n + 2) {

        av_log(s->avctx, AV_LOG_ERROR, "Psot %"PRIu32" too big\n", Psot);

        return AVERROR_INVALIDDATA;

    }



    if (TPsot >= FF_ARRAY_ELEMS(s->tile[Isot].tile_part)) {

        avpriv_request_sample(s->avctx, "Support for %"PRIu8" components", TPsot);

        return AVERROR_PATCHWELCOME;

    }



    s->tile[Isot].tp_idx = TPsot;

    tp             = s->tile[Isot].tile_part + TPsot;

    tp->tile_index = Isot;

    tp->tp_end     = s->g.buffer + Psot - n - 2;



    if (!TPsot) {

        Jpeg2000Tile *tile = s->tile + s->curtileno;



        /* copy defaults */

        memcpy(tile->codsty, s->codsty, s->ncomponents * sizeof(Jpeg2000CodingStyle));

        memcpy(tile->qntsty, s->qntsty, s->ncomponents * sizeof(Jpeg2000QuantStyle));

    }



    return 0;

}
