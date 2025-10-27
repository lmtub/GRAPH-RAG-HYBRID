static int frame_copy_props(AVFrame *dst, const AVFrame *src, int force_copy)

{

    int i;



    dst->key_frame              = src->key_frame;

    dst->pict_type              = src->pict_type;

    dst->sample_aspect_ratio    = src->sample_aspect_ratio;

    dst->pts                    = src->pts;

    dst->repeat_pict            = src->repeat_pict;

    dst->interlaced_frame       = src->interlaced_frame;

    dst->top_field_first        = src->top_field_first;

    dst->palette_has_changed    = src->palette_has_changed;

    dst->sample_rate            = src->sample_rate;

    dst->opaque                 = src->opaque;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

    dst->pkt_pts                = src->pkt_pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    dst->pkt_dts                = src->pkt_dts;

    dst->pkt_pos                = src->pkt_pos;

    dst->pkt_size               = src->pkt_size;

    dst->pkt_duration           = src->pkt_duration;

    dst->reordered_opaque       = src->reordered_opaque;

    dst->quality                = src->quality;

    dst->best_effort_timestamp  = src->best_effort_timestamp;

    dst->coded_picture_number   = src->coded_picture_number;

    dst->display_picture_number = src->display_picture_number;

    dst->flags                  = src->flags;

    dst->decode_error_flags     = src->decode_error_flags;

    dst->color_primaries        = src->color_primaries;

    dst->color_trc              = src->color_trc;

    dst->colorspace             = src->colorspace;

    dst->color_range            = src->color_range;

    dst->chroma_location        = src->chroma_location;



    av_dict_copy(&dst->metadata, src->metadata, 0);



#if FF_API_ERROR_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    memcpy(dst->error, src->error, sizeof(dst->error));

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    for (i = 0; i < src->nb_side_data; i++) {

        const AVFrameSideData *sd_src = src->side_data[i];

        AVFrameSideData *sd_dst;

        if (   sd_src->type == AV_FRAME_DATA_PANSCAN

            && (src->width != dst->width || src->height != dst->height))

            continue;

        if (force_copy) {

            sd_dst = av_frame_new_side_data(dst, sd_src->type,

                                            sd_src->size);

            if (!sd_dst) {

                wipe_side_data(dst);

                return AVERROR(ENOMEM);


            memcpy(sd_dst->data, sd_src->data, sd_src->size);

        } else {

            sd_dst = av_frame_new_side_data(dst, sd_src->type, 0);

            if (!sd_dst) {

                wipe_side_data(dst);

                return AVERROR(ENOMEM);


            if (sd_src->buf) {

            sd_dst->buf = av_buffer_ref(sd_src->buf);

            if (!sd_dst->buf) {

                wipe_side_data(dst);

                return AVERROR(ENOMEM);


            sd_dst->data = sd_dst->buf->data;

            sd_dst->size = sd_dst->buf->size;



        av_dict_copy(&sd_dst->metadata, sd_src->metadata, 0);




#if FF_API_FRAME_QP

FF_DISABLE_DEPRECATION_WARNINGS

    dst->qscale_table = NULL;

    dst->qstride      = 0;

    dst->qscale_type  = 0;

    av_buffer_unref(&dst->qp_table_buf);

    if (src->qp_table_buf) {

        dst->qp_table_buf = av_buffer_ref(src->qp_table_buf);

        if (dst->qp_table_buf) {

            dst->qscale_table = dst->qp_table_buf->data;

            dst->qstride      = src->qstride;

            dst->qscale_type  = src->qscale_type;



FF_ENABLE_DEPRECATION_WARNINGS

#endif



    return 0;
