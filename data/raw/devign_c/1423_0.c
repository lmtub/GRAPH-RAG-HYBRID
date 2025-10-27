static int gif_read_extension(GifState *s)

{

    int ext_code, ext_len, gce_flags, gce_transparent_index;



    /* There must be at least 2 bytes:

     * 1 for extension label and 1 for extension length. */

    if (bytestream2_get_bytes_left(&s->gb) < 2)

        return AVERROR_INVALIDDATA;



    ext_code = bytestream2_get_byteu(&s->gb);

    ext_len = bytestream2_get_byteu(&s->gb);



    av_dlog(s->avctx, "ext_code=0x%x len=%d\n", ext_code, ext_len);



    switch(ext_code) {

    case GIF_GCE_EXT_LABEL:

        if (ext_len != 4)

            goto discard_ext;



        /* We need at least 5 bytes more: 4 is for extension body

         * and 1 for next block size. */

        if (bytestream2_get_bytes_left(&s->gb) < 5)

            return AVERROR_INVALIDDATA;



        gce_flags = bytestream2_get_byteu(&s->gb);

        bytestream2_skipu(&s->gb, 2);    // delay during which the frame is shown

        gce_transparent_index = bytestream2_get_byteu(&s->gb);

        if (gce_flags & 0x01)

            s->transparent_color_index = gce_transparent_index;

        else

            s->transparent_color_index = -1;

        s->gce_disposal = (gce_flags >> 2) & 0x7;



        av_dlog(s->avctx, "gce_flags=%x tcolor=%d disposal=%d\n",

               gce_flags,

               s->transparent_color_index, s->gce_disposal);



        if (s->gce_disposal > 3) {

            s->gce_disposal = GCE_DISPOSAL_NONE;

            av_dlog(s->avctx, "invalid value in gce_disposal (%d). Using default value of 0.\n", ext_len);

        }



        ext_len = bytestream2_get_byteu(&s->gb);

        break;

    }



    /* NOTE: many extension blocks can come after */

 discard_ext:

    while (ext_len != 0) {

        /* There must be at least ext_len bytes and 1 for next block size byte. */

        if (bytestream2_get_bytes_left(&s->gb) < ext_len + 1)

            return AVERROR_INVALIDDATA;



        bytestream2_skipu(&s->gb, ext_len);

        ext_len = bytestream2_get_byteu(&s->gb);



        av_dlog(s->avctx, "ext_len1=%d\n", ext_len);

    }

    return 0;

}
