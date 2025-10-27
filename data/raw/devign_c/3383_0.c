static int gif_read_header1(GifState *s)

{

    uint8_t sig[6];

    int v, n;

    int background_color_index;



    if (bytestream2_get_bytes_left(&s->gb) < 13)

        return AVERROR_INVALIDDATA;



    /* read gif signature */

    bytestream2_get_bufferu(&s->gb, sig, 6);

    if (memcmp(sig, gif87a_sig, 6) != 0 &&

        memcmp(sig, gif89a_sig, 6) != 0)

        return AVERROR_INVALIDDATA;



    /* read screen header */

    s->transparent_color_index = -1;

    s->screen_width = bytestream2_get_le16u(&s->gb);

    s->screen_height = bytestream2_get_le16u(&s->gb);

    if(   (unsigned)s->screen_width  > 32767

       || (unsigned)s->screen_height > 32767){

        av_log(s->avctx, AV_LOG_ERROR, "picture size too large\n");

        return AVERROR_INVALIDDATA;

    }



    av_fast_malloc(&s->idx_line, &s->idx_line_size, s->screen_width);

    if (!s->idx_line)

        return AVERROR(ENOMEM);



    v = bytestream2_get_byteu(&s->gb);

    s->color_resolution = ((v & 0x70) >> 4) + 1;

    s->has_global_palette = (v & 0x80);

    s->bits_per_pixel = (v & 0x07) + 1;

    background_color_index = bytestream2_get_byteu(&s->gb);

    n = bytestream2_get_byteu(&s->gb);

    if (n) {

        s->avctx->sample_aspect_ratio.num = n + 15;

        s->avctx->sample_aspect_ratio.den = 64;

    }



    av_dlog(s->avctx, "screen_w=%d screen_h=%d bpp=%d global_palette=%d\n",

           s->screen_width, s->screen_height, s->bits_per_pixel,

           s->has_global_palette);



    if (s->has_global_palette) {

        s->background_color_index = background_color_index;

        n = 1 << s->bits_per_pixel;

        if (bytestream2_get_bytes_left(&s->gb) < n * 3)

            return AVERROR_INVALIDDATA;



        gif_read_palette(s, s->global_palette, n);

        s->bg_color = s->global_palette[s->background_color_index];

    } else

        s->background_color_index = -1;



    return 0;

}
