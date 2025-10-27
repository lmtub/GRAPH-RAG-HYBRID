int av_packet_copy_props(AVPacket *dst, const AVPacket *src)

{

    int i;



    dst->pts                  = src->pts;

    dst->dts                  = src->dts;

    dst->pos                  = src->pos;

    dst->duration             = src->duration;

    dst->convergence_duration = src->convergence_duration;

    dst->flags                = src->flags;

    dst->stream_index         = src->stream_index;

    dst->side_data_elems      = src->side_data_elems;



    for (i = 0; i < src->side_data_elems; i++) {

         enum AVPacketSideDataType type = src->side_data[i].type;

         int size          = src->side_data[i].size;

         uint8_t *src_data = src->side_data[i].data;

         uint8_t *dst_data = av_packet_new_side_data(dst, type, size);



        if (!dst_data) {

            av_packet_free_side_data(dst);

            return AVERROR(ENOMEM);

        }

        memcpy(dst_data, src_data, size);

    }



    return 0;

}
