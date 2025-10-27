static int h264_export_frame_props(H264Context *h)

{

    const SPS *sps = h->ps.sps;

    H264Picture *cur = h->cur_pic_ptr;



    cur->f->interlaced_frame = 0;

    cur->f->repeat_pict      = 0;



    /* Signal interlacing information externally. */

    /* Prioritize picture timing SEI information over used

     * decoding process if it exists. */



    if (sps->pic_struct_present_flag) {

        H264SEIPictureTiming *pt = &h->sei.picture_timing;

        switch (pt->pic_struct) {

        case SEI_PIC_STRUCT_FRAME:

            break;

        case SEI_PIC_STRUCT_TOP_FIELD:

        case SEI_PIC_STRUCT_BOTTOM_FIELD:

            cur->f->interlaced_frame = 1;

            break;

        case SEI_PIC_STRUCT_TOP_BOTTOM:

        case SEI_PIC_STRUCT_BOTTOM_TOP:

            if (FIELD_OR_MBAFF_PICTURE(h))

                cur->f->interlaced_frame = 1;

            else

                // try to flag soft telecine progressive

                cur->f->interlaced_frame = h->prev_interlaced_frame;

            break;

        case SEI_PIC_STRUCT_TOP_BOTTOM_TOP:

        case SEI_PIC_STRUCT_BOTTOM_TOP_BOTTOM:

            /* Signal the possibility of telecined film externally

             * (pic_struct 5,6). From these hints, let the applications

             * decide if they apply deinterlacing. */

            cur->f->repeat_pict = 1;

            break;

        case SEI_PIC_STRUCT_FRAME_DOUBLING:

            cur->f->repeat_pict = 2;

            break;

        case SEI_PIC_STRUCT_FRAME_TRIPLING:

            cur->f->repeat_pict = 4;

            break;

        }



        if ((pt->ct_type & 3) &&

            pt->pic_struct <= SEI_PIC_STRUCT_BOTTOM_TOP)

            cur->f->interlaced_frame = (pt->ct_type & (1 << 1)) != 0;

    } else {

        /* Derive interlacing flag from used decoding process. */

        cur->f->interlaced_frame = FIELD_OR_MBAFF_PICTURE(h);

    }

    h->prev_interlaced_frame = cur->f->interlaced_frame;



    if (cur->field_poc[0] != cur->field_poc[1]) {

        /* Derive top_field_first from field pocs. */

        cur->f->top_field_first = cur->field_poc[0] < cur->field_poc[1];

    } else {

        if (cur->f->interlaced_frame || sps->pic_struct_present_flag) {

            /* Use picture timing SEI information. Even if it is a

             * information of a past frame, better than nothing. */

            if (h->sei.picture_timing.pic_struct == SEI_PIC_STRUCT_TOP_BOTTOM ||

                h->sei.picture_timing.pic_struct == SEI_PIC_STRUCT_TOP_BOTTOM_TOP)

                cur->f->top_field_first = 1;

            else

                cur->f->top_field_first = 0;

        } else {

            /* Most likely progressive */

            cur->f->top_field_first = 0;

        }

    }



    if (h->sei.frame_packing.present &&

        h->sei.frame_packing.arrangement_type >= 0 &&

        h->sei.frame_packing.arrangement_type <= 6 &&

        h->sei.frame_packing.content_interpretation_type > 0 &&

        h->sei.frame_packing.content_interpretation_type < 3) {

        H264SEIFramePacking *fp = &h->sei.frame_packing;

        AVStereo3D *stereo = av_stereo3d_create_side_data(cur->f);

        if (!stereo)

            return AVERROR(ENOMEM);



        switch (fp->arrangement_type) {

        case 0:

            stereo->type = AV_STEREO3D_CHECKERBOARD;

            break;

        case 1:

            stereo->type = AV_STEREO3D_COLUMNS;

            break;

        case 2:

            stereo->type = AV_STEREO3D_LINES;

            break;

        case 3:

            if (fp->quincunx_subsampling)

                stereo->type = AV_STEREO3D_SIDEBYSIDE_QUINCUNX;

            else

                stereo->type = AV_STEREO3D_SIDEBYSIDE;

            break;

        case 4:

            stereo->type = AV_STEREO3D_TOPBOTTOM;

            break;

        case 5:

            stereo->type = AV_STEREO3D_FRAMESEQUENCE;

            break;

        case 6:

            stereo->type = AV_STEREO3D_2D;

            break;

        }



        if (fp->content_interpretation_type == 2)

            stereo->flags = AV_STEREO3D_FLAG_INVERT;

    }



    if (h->sei.display_orientation.present &&

        (h->sei.display_orientation.anticlockwise_rotation ||

         h->sei.display_orientation.hflip ||

         h->sei.display_orientation.vflip)) {

        H264SEIDisplayOrientation *o = &h->sei.display_orientation;

        double angle = o->anticlockwise_rotation * 360 / (double) (1 << 16);

        AVFrameSideData *rotation = av_frame_new_side_data(cur->f,

                                                           AV_FRAME_DATA_DISPLAYMATRIX,

                                                           sizeof(int32_t) * 9);

        if (!rotation)

            return AVERROR(ENOMEM);



        av_display_rotation_set((int32_t *)rotation->data, angle);

        av_display_matrix_flip((int32_t *)rotation->data,

                               o->hflip, o->vflip);

    }



    if (h->sei.afd.present) {

        AVFrameSideData *sd = av_frame_new_side_data(cur->f, AV_FRAME_DATA_AFD,

                                                     sizeof(uint8_t));

        if (!sd)

            return AVERROR(ENOMEM);



        *sd->data = h->sei.afd.active_format_description;

        h->sei.afd.present = 0;

    }



    if (h->sei.a53_caption.a53_caption) {

        H264SEIA53Caption *a53 = &h->sei.a53_caption;

        AVFrameSideData *sd = av_frame_new_side_data(cur->f,

                                                     AV_FRAME_DATA_A53_CC,

                                                     a53->a53_caption_size);

        if (!sd)

            return AVERROR(ENOMEM);



        memcpy(sd->data, a53->a53_caption, a53->a53_caption_size);

        av_freep(&a53->a53_caption);

        a53->a53_caption_size = 0;

    }



    return 0;

}
