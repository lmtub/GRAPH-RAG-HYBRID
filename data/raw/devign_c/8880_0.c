static void set_stream_info_from_input_stream(AVStream *st, struct playlist *pls, AVStream *ist)

{

    avcodec_parameters_copy(st->codecpar, ist->codecpar);



    if (pls->is_id3_timestamped) /* custom timestamps via id3 */

        avpriv_set_pts_info(st, 33, 1, MPEG_TIME_BASE);

    else

        avpriv_set_pts_info(st, ist->pts_wrap_bits, ist->time_base.num, ist->time_base.den);



    st->internal->need_context_update = 1;

}
