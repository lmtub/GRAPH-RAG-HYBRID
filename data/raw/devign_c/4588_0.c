void avfilter_draw_slice(AVFilterLink *link, int y, int h)

{

    uint8_t *src[4], *dst[4];

    int i, j, hsub, vsub;



    /* copy the slice if needed for permission reasons */

    if(link->srcpic) {

        avcodec_get_chroma_sub_sample(link->format, &hsub, &vsub);



        src[0] = link->srcpic-> data[0] + y * link->srcpic-> linesize[0];

        dst[0] = link->cur_pic->data[0] + y * link->cur_pic->linesize[0];

        for(i = 1; i < 4; i ++) {

            if(link->srcpic->data[i]) {

                src[i] = link->srcpic-> data[i] + (y >> vsub) * link->srcpic-> linesize[i];

                dst[i] = link->cur_pic->data[i] + (y >> vsub) * link->cur_pic->linesize[i];

            } else

                src[i] = dst[i] = NULL;

        }

        for(j = 0; j < h; j ++) {

            memcpy(dst[0], src[0], link->cur_pic->linesize[0]);

            src[0] += link->srcpic ->linesize[0];

            dst[0] += link->cur_pic->linesize[0];

        }

        for(i = 1; i < 4; i ++) {

            if(!src[i]) continue;



            for(j = 0; j < h >> vsub; j ++) {

                memcpy(dst[i], src[i], link->cur_pic->linesize[i]);

                src[i] += link->srcpic ->linesize[i];

                dst[i] += link->cur_pic->linesize[i];

            }

        }

    }



    if(!link_dpad(link).draw_slice)

        return;



    link_dpad(link).draw_slice(link, y, h);

}
