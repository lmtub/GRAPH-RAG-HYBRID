static int dirac_decode_picture_header(DiracContext *s)

{

    unsigned retire, picnum;

    int i, j, ret;

    int64_t refdist, refnum;

    GetBitContext *gb = &s->gb;



    /* [DIRAC_STD] 11.1.1 Picture Header. picture_header() PICTURE_NUM */

    picnum = s->current_picture->avframe->display_picture_number = get_bits_long(gb, 32);





    av_log(s->avctx,AV_LOG_DEBUG,"PICTURE_NUM: %d\n",picnum);



    /* if this is the first keyframe after a sequence header, start our

       reordering from here */

    if (s->frame_number < 0)

        s->frame_number = picnum;



    s->ref_pics[0] = s->ref_pics[1] = NULL;

    for (i = 0; i < s->num_refs; i++) {

        refnum = (picnum + dirac_get_se_golomb(gb)) & 0xFFFFFFFF;

        refdist = INT64_MAX;



        /* find the closest reference to the one we want */

        /* Jordi: this is needed if the referenced picture hasn't yet arrived */

        for (j = 0; j < MAX_REFERENCE_FRAMES && refdist; j++)

            if (s->ref_frames[j]

                && FFABS(s->ref_frames[j]->avframe->display_picture_number - refnum) < refdist) {

                s->ref_pics[i] = s->ref_frames[j];

                refdist = FFABS(s->ref_frames[j]->avframe->display_picture_number - refnum);

            }



        if (!s->ref_pics[i] || refdist)

            av_log(s->avctx, AV_LOG_DEBUG, "Reference not found\n");



        /* if there were no references at all, allocate one */

        if (!s->ref_pics[i])

            for (j = 0; j < MAX_FRAMES; j++)

                if (!s->all_frames[j].avframe->data[0]) {

                    s->ref_pics[i] = &s->all_frames[j];

                    get_buffer_with_edge(s->avctx, s->ref_pics[i]->avframe, AV_GET_BUFFER_FLAG_REF);

                    break;

                }



        if (!s->ref_pics[i]) {

            av_log(s->avctx, AV_LOG_ERROR, "Reference could not be allocated\n");

            return AVERROR_INVALIDDATA;

        }



    }



    /* retire the reference frames that are not used anymore */

    if (s->current_picture->reference) {

        retire = (picnum + dirac_get_se_golomb(gb)) & 0xFFFFFFFF;

        if (retire != picnum) {

            DiracFrame *retire_pic = remove_frame(s->ref_frames, retire);



            if (retire_pic)

                retire_pic->reference &= DELAYED_PIC_REF;

            else

                av_log(s->avctx, AV_LOG_DEBUG, "Frame to retire not found\n");

        }



        /* if reference array is full, remove the oldest as per the spec */

        while (add_frame(s->ref_frames, MAX_REFERENCE_FRAMES, s->current_picture)) {

            av_log(s->avctx, AV_LOG_ERROR, "Reference frame overflow\n");

            remove_frame(s->ref_frames, s->ref_frames[0]->avframe->display_picture_number)->reference &= DELAYED_PIC_REF;

        }

    }



    if (s->num_refs) {

        ret = dirac_unpack_prediction_parameters(s);  /* [DIRAC_STD] 11.2 Picture Prediction Data. picture_prediction() */

        if (ret < 0)

            return ret;

        ret = dirac_unpack_block_motion_data(s);      /* [DIRAC_STD] 12. Block motion data syntax                       */

        if (ret < 0)

            return ret;

    }

    ret = dirac_unpack_idwt_params(s);                /* [DIRAC_STD] 11.3 Wavelet transform data                        */

    if (ret < 0)

        return ret;



    init_planes(s);

    return 0;

}
