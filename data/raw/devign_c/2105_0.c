static av_cold int hevc_decode_free(AVCodecContext *avctx)

{

    HEVCContext       *s = avctx->priv_data;

    HEVCLocalContext *lc = s->HEVClc;

    int i;



    pic_arrays_free(s);



    av_freep(&s->md5_ctx);



    for(i=0; i < s->nals_allocated; i++) {

        av_freep(&s->skipped_bytes_pos_nal[i]);

    }

    av_freep(&s->skipped_bytes_pos_size_nal);

    av_freep(&s->skipped_bytes_nal);

    av_freep(&s->skipped_bytes_pos_nal);



    av_freep(&s->cabac_state);



    av_frame_free(&s->tmp_frame);

    av_frame_free(&s->output_frame);



    for (i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {

        ff_hevc_unref_frame(s, &s->DPB[i], ~0);

        av_frame_free(&s->DPB[i].frame);

    }



    for (i = 0; i < FF_ARRAY_ELEMS(s->vps_list); i++)

        av_buffer_unref(&s->vps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(s->sps_list); i++)

        av_buffer_unref(&s->sps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(s->pps_list); i++)

        av_buffer_unref(&s->pps_list[i]);

    s->sps = NULL;

    s->pps = NULL;

    s->vps = NULL;



    av_buffer_unref(&s->current_sps);



    av_freep(&s->sh.entry_point_offset);

    av_freep(&s->sh.offset);

    av_freep(&s->sh.size);



    for (i = 1; i < s->threads_number; i++) {

        lc = s->HEVClcList[i];

        if (lc) {

            av_freep(&s->HEVClcList[i]);

            av_freep(&s->sList[i]);

        }

    }

    if (s->HEVClc == s->HEVClcList[0])

        s->HEVClc = NULL;

    av_freep(&s->HEVClcList[0]);



    for (i = 0; i < s->nals_allocated; i++)

        av_freep(&s->nals[i].rbsp_buffer);

    av_freep(&s->nals);

    s->nals_allocated = 0;



    return 0;

}
