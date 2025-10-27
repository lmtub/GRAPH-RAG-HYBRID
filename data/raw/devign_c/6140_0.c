void ff_mpv_frame_end(MpegEncContext *s)

{

#if FF_API_XVMC

FF_DISABLE_DEPRECATION_WARNINGS

    /* redraw edges for the frame if decoding didn't complete */

    // just to make sure that all data is rendered.

    if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration) {

        ff_xvmc_field_end(s);

    } else

FF_ENABLE_DEPRECATION_WARNINGS

#endif /* FF_API_XVMC */



    emms_c();



    if (s->current_picture.reference)

        ff_thread_report_progress(&s->current_picture_ptr->tf, INT_MAX, 0);

}
