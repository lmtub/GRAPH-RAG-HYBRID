static int output_frame(H264Context *h, AVFrame *dst, AVFrame *src)

{

    int i;

    int ret = av_frame_ref(dst, src);

    if (ret < 0)

        return ret;



    if (!h->sps.crop)

        return 0;



    for (i = 0; i < 3; i++) {

        int hshift = (i > 0) ? h->chroma_x_shift : 0;

        int vshift = (i > 0) ? h->chroma_y_shift : 0;

        int off    = ((h->sps.crop_left >> hshift) << h->pixel_shift) +

            (h->sps.crop_top  >> vshift) * dst->linesize[i];

        dst->data[i] += off;

    }

    return 0;

}
