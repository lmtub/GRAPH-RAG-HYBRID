static int parse_presentation_segment(AVCodecContext *avctx,

                                      const uint8_t *buf, int buf_size,

                                      int64_t pts)

{

    PGSSubContext *ctx = avctx->priv_data;



    int x, y, ret;



    int w = bytestream_get_be16(&buf);

    int h = bytestream_get_be16(&buf);



    ctx->presentation.pts = pts;



    av_dlog(avctx, "Video Dimensions %dx%d\n",

            w, h);

    ret = ff_set_dimensions(avctx, w, h);

    if (ret < 0)

        return ret;



    /* Skip 1 bytes of unknown, frame rate? */

    buf++;



    ctx->presentation.id_number = bytestream_get_be16(&buf);



    /*

     * Skip 3 bytes of unknown:

     *     state

     *     palette_update_flag (0x80),

     *     palette_id_to_use,

     */

    buf += 3;



    ctx->presentation.object_number = bytestream_get_byte(&buf);

    ctx->presentation.composition_flag = 0;

    if (!ctx->presentation.object_number)

        return 0;



    /*

     * Skip 3 bytes of unknown:

     *     object_id_ref (2 bytes),

     *     window_id_ref,

     */

    buf += 3;

    ctx->presentation.composition_flag = bytestream_get_byte(&buf);



    x = bytestream_get_be16(&buf);

    y = bytestream_get_be16(&buf);



    /* TODO If cropping, cropping_x, cropping_y, cropping_width, cropping_height (all 2 bytes).*/



    av_dlog(avctx, "Subtitle Placement x=%d, y=%d\n", x, y);



    if (x > avctx->width || y > avctx->height) {

        av_log(avctx, AV_LOG_ERROR, "Subtitle out of video bounds. x = %d, y = %d, video width = %d, video height = %d.\n",

               x, y, avctx->width, avctx->height);

        x = 0; y = 0;

    }



    /* Fill in dimensions */

    ctx->presentation.x = x;

    ctx->presentation.y = y;



    return 0;

}
