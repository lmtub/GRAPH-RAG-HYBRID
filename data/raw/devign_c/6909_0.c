static void dvbsub_parse_display_definition_segment(AVCodecContext *avctx,

                                                    const uint8_t *buf,

                                                    int buf_size)

{

    DVBSubContext *ctx = avctx->priv_data;

    DVBSubDisplayDefinition *display_def = ctx->display_definition;

    int dds_version, info_byte;



    if (buf_size < 5)

        return;



    info_byte   = bytestream_get_byte(&buf);

    dds_version = info_byte >> 4;

    if (display_def && display_def->version == dds_version)

        return; // already have this display definition version



    if (!display_def) {

        display_def             = av_mallocz(sizeof(*display_def));

        ctx->display_definition = display_def;

    }

    if (!display_def)

        return;



    display_def->version = dds_version;

    display_def->x       = 0;

    display_def->y       = 0;

    display_def->width   = bytestream_get_be16(&buf) + 1;

    display_def->height  = bytestream_get_be16(&buf) + 1;



    if (buf_size < 13)

        return;



    if (info_byte & 1<<3) { // display_window_flag

        display_def->x = bytestream_get_be16(&buf);

        display_def->y = bytestream_get_be16(&buf);

        display_def->width  = bytestream_get_be16(&buf) - display_def->x + 1;

        display_def->height = bytestream_get_be16(&buf) - display_def->y + 1;

    }

}
