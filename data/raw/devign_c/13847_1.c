void av_image_copy(uint8_t *dst_data[4], int dst_linesizes[4],

                   const uint8_t *src_data[4], const int src_linesizes[4],

                   enum AVPixelFormat pix_fmt, int width, int height)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);



    if (!desc || desc->flags & PIX_FMT_HWACCEL)




    if (desc->flags & PIX_FMT_PAL ||

        desc->flags & PIX_FMT_PSEUDOPAL) {

        av_image_copy_plane(dst_data[0], dst_linesizes[0],

                            src_data[0], src_linesizes[0],

                            width, height);

        /* copy the palette */

        memcpy(dst_data[1], src_data[1], 4*256);

    } else {

        int i, planes_nb = 0;



        for (i = 0; i < desc->nb_components; i++)

            planes_nb = FFMAX(planes_nb, desc->comp[i].plane + 1);



        for (i = 0; i < planes_nb; i++) {

            int h = height;

            int bwidth = av_image_get_linesize(pix_fmt, width, i);





            if (i == 1 || i == 2) {

                h= -((-height)>>desc->log2_chroma_h);


            av_image_copy_plane(dst_data[i], dst_linesizes[i],

                                src_data[i], src_linesizes[i],

                                bwidth, h);


