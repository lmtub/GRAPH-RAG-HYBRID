static int roq_encode_video(RoqContext *enc)

{

    RoqTempdata *tempData = enc->tmpData;

    int i, ret;



    memset(tempData, 0, sizeof(*tempData));



    ret = create_cel_evals(enc, tempData);

    if (ret < 0)

        return ret;



    ret = generate_new_codebooks(enc, tempData);

    if (ret < 0)

        return ret;



    if (enc->framesSinceKeyframe >= 1) {

        motion_search(enc, 8);

        motion_search(enc, 4);

    }



 retry_encode:

    for (i=0; i<enc->width*enc->height/64; i++)

        gather_data_for_cel(tempData->cel_evals + i, enc, tempData);



    /* Quake 3 can't handle chunks bigger than 65535 bytes */

    if (tempData->mainChunkSize/8 > 65535) {

        av_log(enc->avctx, AV_LOG_ERROR,

               "Warning, generated a frame too big (%d > 65535), "

               "try using a smaller qscale value.\n",

               tempData->mainChunkSize/8);

        enc->lambda *= 1.5;

        tempData->mainChunkSize = 0;

        memset(tempData->used_option, 0, sizeof(tempData->used_option));

        memset(tempData->codebooks.usedCB4, 0,

               sizeof(tempData->codebooks.usedCB4));

        memset(tempData->codebooks.usedCB2, 0,

               sizeof(tempData->codebooks.usedCB2));



        goto retry_encode;

    }



    remap_codebooks(enc, tempData);



    write_codebooks(enc, tempData);



    reconstruct_and_encode_image(enc, tempData, enc->width, enc->height,

                                 enc->width*enc->height/64);



    enc->avctx->coded_frame = enc->current_frame;



    /* Rotate frame history */

    FFSWAP(AVFrame *, enc->current_frame, enc->last_frame);

    FFSWAP(motion_vect *, enc->last_motion4, enc->this_motion4);

    FFSWAP(motion_vect *, enc->last_motion8, enc->this_motion8);



    av_free(tempData->cel_evals);

    av_free(tempData->closest_cb2);



    enc->framesSinceKeyframe++;



    return 0;

}
