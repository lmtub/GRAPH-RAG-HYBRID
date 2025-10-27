static int output_frame(H264Context *h, AVFrame *dst, H264Picture *srcp)

{

    AVFrame *src = srcp->f;

    int ret;



    if (src->format == AV_PIX_FMT_VIDEOTOOLBOX && src->buf[0]->size == 1)

        return AVERROR_EXTERNAL;



    ret = av_frame_ref(dst, src);

    if (ret < 0)

        return ret;



    av_dict_set(&dst->metadata, "stereo_mode", ff_h264_sei_stereo_mode(&h->sei.frame_packing), 0);



    if (srcp->sei_recovery_frame_cnt == 0)

        dst->key_frame = 1;



    return 0;

}
