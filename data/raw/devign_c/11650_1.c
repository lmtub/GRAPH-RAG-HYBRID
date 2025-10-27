static int ipvideo_decode_frame(AVCodecContext *avctx,
                                void *data, int *got_frame,
                                AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    IpvideoContext *s = avctx->priv_data;
    AVFrame *frame = data;
    int ret;
    int send_buffer;
    int frame_format;
    int video_data_size;
    if (av_packet_get_side_data(avpkt, AV_PKT_DATA_PARAM_CHANGE, NULL)) {
        av_frame_unref(s->last_frame);
        av_frame_unref(s->second_last_frame);
    if (buf_size < 8)
        return AVERROR_INVALIDDATA;
    frame_format         = AV_RL8(buf);
    send_buffer          = AV_RL8(buf + 1);
    video_data_size      = AV_RL16(buf + 2);
    s->decoding_map_size = AV_RL16(buf + 4);
    s->skip_map_size     = AV_RL16(buf + 6);
    switch(frame_format) {
        case 0x06:
            if (s->decoding_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Decoding map for format 0x06\n");
                return AVERROR_INVALIDDATA;
            if (s->skip_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Skip map for format 0x06\n");
                return AVERROR_INVALIDDATA;
            if (s->is_16bpp) {
                av_log(avctx, AV_LOG_ERROR, "Video format 0x06 does not support 16bpp movies\n");
                return AVERROR_INVALIDDATA;
            /* Decoding map for 0x06 frame format is at the top of pixeldata */
            s->decoding_map_size = ((s->avctx->width / 8) * (s->avctx->height / 8)) * 2;
            s->decoding_map = buf + 8 + 14; /* 14 bits of op data */
            video_data_size -= s->decoding_map_size + 14;
            if (video_data_size <= 0)
                return AVERROR_INVALIDDATA;
            if (buf_size < 8 + s->decoding_map_size + 14 + video_data_size)
                return AVERROR_INVALIDDATA;
            bytestream2_init(&s->stream_ptr, buf + 8 + s->decoding_map_size + 14, video_data_size);
            break;
        case 0x10:
            if (! s->decoding_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Empty decoding map for format 0x10\n");
                return AVERROR_INVALIDDATA;
            if (! s->skip_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Empty skip map for format 0x10\n");
                return AVERROR_INVALIDDATA;
            if (s->is_16bpp) {
                av_log(avctx, AV_LOG_ERROR, "Video format 0x10 does not support 16bpp movies\n");
                return AVERROR_INVALIDDATA;
            if (buf_size < 8 + video_data_size + s->decoding_map_size + s->skip_map_size)
                return AVERROR_INVALIDDATA;
            bytestream2_init(&s->stream_ptr, buf + 8, video_data_size);
            s->decoding_map = buf + 8 + video_data_size;
            s->skip_map = buf + 8 + video_data_size + s->decoding_map_size;
            break;
        case 0x11:
            if (! s->decoding_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Empty decoding map for format 0x11\n");
                return AVERROR_INVALIDDATA;
            if (s->skip_map_size) {
                av_log(avctx, AV_LOG_ERROR, "Skip map for format 0x11\n");
                return AVERROR_INVALIDDATA;
            if (buf_size < 8 + video_data_size + s->decoding_map_size)
                return AVERROR_INVALIDDATA;
            bytestream2_init(&s->stream_ptr, buf + 8, video_data_size);
            s->decoding_map = buf + 8 + video_data_size;
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Frame type 0x%02X unsupported\n", frame_format);
    /* ensure we can't overread the packet */
    if (buf_size < 8 + s->decoding_map_size + video_data_size + s->skip_map_size) {
        av_log(avctx, AV_LOG_ERROR, "Invalid IP packet size\n");
        return AVERROR_INVALIDDATA;
    if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)
    if (!s->is_16bpp) {
        int size;
        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, &size);
        if (pal && size == AVPALETTE_SIZE) {
            frame->palette_has_changed = 1;
            memcpy(s->pal, pal, AVPALETTE_SIZE);
        } else if (pal) {
            av_log(avctx, AV_LOG_ERROR, "Palette size %d is wrong\n", size);
    switch(frame_format) {
        case 0x06:
            ipvideo_decode_format_06_opcodes(s, frame);
            break;
        case 0x10:
            ipvideo_decode_format_10_opcodes(s, frame);
            break;
        case 0x11:
            ipvideo_decode_format_11_opcodes(s, frame);
            break;
    *got_frame = send_buffer;
    /* shuffle frames */
    av_frame_unref(s->second_last_frame);
    FFSWAP(AVFrame*, s->second_last_frame, s->last_frame);
    if ((ret = av_frame_ref(s->last_frame, frame)) < 0)
    /* report that the buffer was completely consumed */
    return buf_size;