static void dvbsub_parse_object_segment(AVCodecContext *avctx,

                                        const uint8_t *buf, int buf_size)

{

    DVBSubContext *ctx = avctx->priv_data;



    const uint8_t *buf_end = buf + buf_size;

    const uint8_t *block;

    int object_id;

    DVBSubObject *object;

    DVBSubObjectDisplay *display;

    int top_field_len, bottom_field_len;



    int coding_method, non_modifying_color;



    object_id = AV_RB16(buf);

    buf += 2;



    object = get_object(ctx, object_id);



    if (!object)

        return;



    coding_method = ((*buf) >> 2) & 3;

    non_modifying_color = ((*buf++) >> 1) & 1;



    if (coding_method == 0) {

        top_field_len = AV_RB16(buf);

        buf += 2;

        bottom_field_len = AV_RB16(buf);

        buf += 2;



        if (buf + top_field_len + bottom_field_len > buf_end) {

            av_log(avctx, AV_LOG_ERROR, "Field data size too large\n");

            return;

        }



        for (display = object->display_list; display; display = display->object_list_next) {

            block = buf;



            dvbsub_parse_pixel_data_block(avctx, display, block, top_field_len, 0,

                                            non_modifying_color);



            if (bottom_field_len > 0)

                block = buf + top_field_len;

            else

                bottom_field_len = top_field_len;



            dvbsub_parse_pixel_data_block(avctx, display, block, bottom_field_len, 1,

                                            non_modifying_color);

        }



/*  } else if (coding_method == 1) {*/



    } else {

        av_log(avctx, AV_LOG_ERROR, "Unknown object coding %d\n", coding_method);

    }



}
