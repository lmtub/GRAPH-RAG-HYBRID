static int decode_user_data(Mpeg4DecContext *ctx, GetBitContext *gb)

{

    MpegEncContext *s = &ctx->m;

    char buf[256];

    int i;

    int e;

    int ver = 0, build = 0, ver2 = 0, ver3 = 0;

    char last;



    for (i = 0; i < 255 && get_bits_count(gb) < gb->size_in_bits; i++) {

        if (show_bits(gb, 23) == 0)

            break;

        buf[i] = get_bits(gb, 8);

    }

    buf[i] = 0;



    /* divx detection */

    e = sscanf(buf, "DivX%dBuild%d%c", &ver, &build, &last);

    if (e < 2)

        e = sscanf(buf, "DivX%db%d%c", &ver, &build, &last);

    if (e >= 2) {

        ctx->divx_version = ver;

        ctx->divx_build   = build;

        s->divx_packed  = e == 3 && last == 'p';

    }



    /* libavcodec detection */

    e = sscanf(buf, "FFmpe%*[^b]b%d", &build) + 3;

    if (e != 4)

        e = sscanf(buf, "FFmpeg v%d.%d.%d / libavcodec build: %d", &ver, &ver2, &ver3, &build);

    if (e != 4) {

        e = sscanf(buf, "Lavc%d.%d.%d", &ver, &ver2, &ver3) + 1;

        if (e > 1) {

            if (ver > 0xFF || ver2 > 0xFF || ver3 > 0xFF) {

                av_log(s->avctx, AV_LOG_WARNING,

                     "Unknown Lavc version string encountered, %d.%d.%d; "

                     "clamping sub-version values to 8-bits.\n",

                     ver, ver2, ver3);

            }

            build = ((ver & 0xFF) << 16) + ((ver2 & 0xFF) << 8) + (ver3 & 0xFF);

        }

    }

    if (e != 4) {

        if (strcmp(buf, "ffmpeg") == 0)

            ctx->lavc_build = 4600;

    }

    if (e == 4)

        ctx->lavc_build = build;



    /* Xvid detection */

    e = sscanf(buf, "XviD%d", &build);

    if (e == 1)

        ctx->xvid_build = build;



    return 0;

}
