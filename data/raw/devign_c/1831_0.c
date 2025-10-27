static void roq_encode_video(RoqContext *enc)

{

    RoqTempdata *tempData = enc->tmpData;

    int i;



    memset(tempData, 0, sizeof(*tempData));



    create_cel_evals(enc, tempData);



    generate_new_codebooks(enc, tempData);



    if (enc->framesSinceKeyframe >= 1) {

        motion_search(enc, 8);

        motion_search(enc, 4);

    }



 retry_encode:

    for (i=0; i<enc->width*enc->height/64; i++)

        gather_data_for_cel(tempData->cel_evals + i, enc, tempData);



    /* Quake 3 can't handle chunks bigger than 65536 bytes */

    if (tempData->mainChunkSize/8 > 65536) {

        enc->lambda *= .8;

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

}
