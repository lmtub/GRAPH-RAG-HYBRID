void MPV_common_end(MpegEncContext *s)

{

    int i;



    av_freep(&s->mb_type);

    av_freep(&s->p_mv_table);

    av_freep(&s->b_forw_mv_table);

    av_freep(&s->b_back_mv_table);

    av_freep(&s->b_bidir_forw_mv_table);

    av_freep(&s->b_bidir_back_mv_table);

    av_freep(&s->b_direct_mv_table);

    av_freep(&s->motion_val);

    av_freep(&s->dc_val[0]);

    av_freep(&s->ac_val[0]);

    av_freep(&s->coded_block);

    av_freep(&s->mbintra_table);

    av_freep(&s->cbp_table);

    av_freep(&s->pred_dir_table);

    av_freep(&s->me.scratchpad);

    av_freep(&s->me.map);

    av_freep(&s->me.score_map);

    

    av_freep(&s->mbskip_table);

    av_freep(&s->bitstream_buffer);

    av_freep(&s->tex_pb_buffer);

    av_freep(&s->pb2_buffer);

    av_freep(&s->edge_emu_buffer);

    av_freep(&s->co_located_type_table);

    av_freep(&s->field_mv_table);

    av_freep(&s->field_select_table);

    av_freep(&s->avctx->stats_out);

    av_freep(&s->ac_stats);

    av_freep(&s->error_status_table);



    for(i=0; i<MAX_PICTURE_COUNT; i++){

        free_picture(s, &s->picture[i]);

    }

    s->context_initialized = 0;

}
