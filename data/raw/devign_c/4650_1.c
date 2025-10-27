int av_packet_ref(AVPacket *dst, const AVPacket *src)

{

    int ret;



    ret = av_packet_copy_props(dst, src);

    if (ret < 0)

        return ret;



    if (!src->buf) {

        ret = packet_alloc(&dst->buf, src->size);

        if (ret < 0)

            goto fail;

        memcpy(dst->buf->data, src->data, src->size);



        dst->data = dst->buf->data;

    } else {

        dst->buf = av_buffer_ref(src->buf);

        if (!dst->buf) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        dst->data = src->data;

    }



    dst->size = src->size;



    return 0;

fail:

    av_packet_free_side_data(dst);

    return ret;

}
