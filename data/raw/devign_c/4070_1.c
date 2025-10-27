static void write_frame(AVFormatContext *s, AVPacket *pkt, OutputStream *ost)

{

    AVBitStreamFilterContext *bsfc = ost->bitstream_filters;

    AVCodecContext          *avctx = ost->st->codec;

    int ret;



    if ((avctx->codec_type == AVMEDIA_TYPE_VIDEO && video_sync_method == VSYNC_DROP) ||

        (avctx->codec_type == AVMEDIA_TYPE_AUDIO && audio_sync_method < 0))

        pkt->pts = pkt->dts = AV_NOPTS_VALUE;



    if (avctx->codec_type == AVMEDIA_TYPE_AUDIO && pkt->dts != AV_NOPTS_VALUE) {

        int64_t max = ost->st->cur_dts + !(s->oformat->flags & AVFMT_TS_NONSTRICT);

        if (ost->st->cur_dts && ost->st->cur_dts != AV_NOPTS_VALUE &&  max > pkt->dts) {

            av_log(s, max - pkt->dts > 2 ? AV_LOG_WARNING : AV_LOG_DEBUG, "Audio timestamp %"PRId64" < %"PRId64" invalid, cliping\n", pkt->dts, max);

            pkt->pts = pkt->dts = max;





    /*

     * Audio encoders may split the packets --  #frames in != #packets out.

     * But there is no reordering, so we can limit the number of output packets

     * by simply dropping them here.

     * Counting encoded video frames needs to be done separately because of

     * reordering, see do_video_out()

     */

    if (!(avctx->codec_type == AVMEDIA_TYPE_VIDEO && avctx->codec)) {

        if (ost->frame_number >= ost->max_frames) {

            av_free_packet(pkt);

            return;


        ost->frame_number++;




    while (bsfc) {

        AVPacket new_pkt = *pkt;

        int a = av_bitstream_filter_filter(bsfc, avctx, NULL,

                                           &new_pkt.data, &new_pkt.size,

                                           pkt->data, pkt->size,

                                           pkt->flags & AV_PKT_FLAG_KEY);











        if (a > 0) {

            av_free_packet(pkt);

            new_pkt.destruct = av_destruct_packet;

        } else if (a < 0) {

            av_log(NULL, AV_LOG_ERROR, "Failed to open bitstream filter %s for stream %d with codec %s",

                   bsfc->filter->name, pkt->stream_index,

                   avctx->codec ? avctx->codec->name : "copy");

            print_error("", a);

            if (exit_on_error)

                exit_program(1);


        *pkt = new_pkt;



        bsfc = bsfc->next;




    pkt->stream_index = ost->index;

    ret = av_interleaved_write_frame(s, pkt);

    if (ret < 0) {

        print_error("av_interleaved_write_frame()", ret);

        exit_program(1);

