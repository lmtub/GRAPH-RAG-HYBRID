static int get_siz(Jpeg2000DecoderContext *s)

{

    int i;



    if (bytestream2_get_bytes_left(&s->g) < 36)

        return AVERROR(EINVAL);



    s->avctx->profile = bytestream2_get_be16u(&s->g); // Rsiz

    s->width          = bytestream2_get_be32u(&s->g); // Width

    s->height         = bytestream2_get_be32u(&s->g); // Height

    s->image_offset_x = bytestream2_get_be32u(&s->g); // X0Siz

    s->image_offset_y = bytestream2_get_be32u(&s->g); // Y0Siz

    s->tile_width     = bytestream2_get_be32u(&s->g); // XTSiz

    s->tile_height    = bytestream2_get_be32u(&s->g); // YTSiz

    s->tile_offset_x  = bytestream2_get_be32u(&s->g); // XT0Siz

    s->tile_offset_y  = bytestream2_get_be32u(&s->g); // YT0Siz

    s->ncomponents    = bytestream2_get_be16u(&s->g); // CSiz



    if (s->ncomponents <= 0 || s->ncomponents > 4) {

        av_log(s->avctx, AV_LOG_ERROR, "unsupported/invalid ncomponents: %d\n", s->ncomponents);

        return AVERROR(EINVAL);

    }

    if (s->tile_width<=0 || s->tile_height<=0)

        return AVERROR(EINVAL);



    if (bytestream2_get_bytes_left(&s->g) < 3 * s->ncomponents)

        return AVERROR(EINVAL);



    for (i = 0; i < s->ncomponents; i++) { // Ssiz_i XRsiz_i, YRsiz_i

        uint8_t x = bytestream2_get_byteu(&s->g);

        s->cbps[i]   = (x & 0x7f) + 1;

        s->precision = FFMAX(s->cbps[i], s->precision);

        s->sgnd[i]   = !!(x & 0x80);

        s->cdx[i]    = bytestream2_get_byteu(&s->g);

        s->cdy[i]    = bytestream2_get_byteu(&s->g);

        if (s->cdx[i] != 1 || s->cdy[i] != 1) {

            av_log(s->avctx, AV_LOG_ERROR, "unsupported/ CDxy values %d %d for component %d\n", s->cdx[i], s->cdy[i], i);

            if (!s->cdx[i] || !s->cdy[i])

                return AVERROR_INVALIDDATA;

        }

    }



    s->numXtiles = ff_jpeg2000_ceildiv(s->width  - s->tile_offset_x, s->tile_width);

    s->numYtiles = ff_jpeg2000_ceildiv(s->height - s->tile_offset_y, s->tile_height);



    if (s->numXtiles * (uint64_t)s->numYtiles > INT_MAX/sizeof(Jpeg2000Tile))

        return AVERROR(EINVAL);



    s->tile = av_mallocz(s->numXtiles * s->numYtiles * sizeof(*s->tile));

    if (!s->tile)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->numXtiles * s->numYtiles; i++) {

        Jpeg2000Tile *tile = s->tile + i;



        tile->comp = av_mallocz(s->ncomponents * sizeof(*tile->comp));

        if (!tile->comp)

            return AVERROR(ENOMEM);

    }



    /* compute image size with reduction factor */

    s->avctx->width  = ff_jpeg2000_ceildivpow2(s->width  - s->image_offset_x,

                                               s->reduction_factor);

    s->avctx->height = ff_jpeg2000_ceildivpow2(s->height - s->image_offset_y,

                                               s->reduction_factor);



    switch(s->ncomponents) {

    case 1:

        if (s->precision > 8)

            s->avctx->pix_fmt = AV_PIX_FMT_GRAY16;

        else

            s->avctx->pix_fmt = AV_PIX_FMT_GRAY8;

        break;

    case 3:

        switch (s->avctx->profile) {

        case FF_PROFILE_JPEG2000_DCINEMA_2K:

        case FF_PROFILE_JPEG2000_DCINEMA_4K:

            /* XYZ color-space for digital cinema profiles */

            s->avctx->pix_fmt = AV_PIX_FMT_XYZ12;

            break;

        default:

            if (s->precision > 8)

                s->avctx->pix_fmt = AV_PIX_FMT_RGB48;

            else

                s->avctx->pix_fmt = AV_PIX_FMT_RGB24;

            break;

        }

        break;

    case 4:

        s->avctx->pix_fmt = AV_PIX_FMT_RGBA;

        break;

    default:

        /* pixel format can not be identified */

        s->avctx->pix_fmt = AV_PIX_FMT_NONE;

        break;

    }

    return 0;

}
