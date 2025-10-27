static int brpix_decode_frame(AVCodecContext *avctx,
                              void *data, int *got_frame,
                              AVPacket *avpkt)
{
    BRPixContext *s = avctx->priv_data;
    AVFrame *frame_out = data;
    int ret;
    GetByteContext gb;
    unsigned int bytes_pp;
    unsigned int magic[4];
    unsigned int chunk_type;
    unsigned int data_len;
    BRPixHeader hdr;
    bytestream2_init(&gb, avpkt->data, avpkt->size);
    magic[0] = bytestream2_get_be32(&gb);
    magic[1] = bytestream2_get_be32(&gb);
    magic[2] = bytestream2_get_be32(&gb);
    magic[3] = bytestream2_get_be32(&gb);
    if (magic[0] != 0x12 ||
        magic[1] != 0x8 ||
        magic[2] != 0x2 ||
        magic[3] != 0x2) {
        av_log(avctx, AV_LOG_ERROR, "Not a BRender PIX file\n");
        return AVERROR_INVALIDDATA;
    chunk_type = bytestream2_get_be32(&gb);
    if (chunk_type != 0x3 && chunk_type != 0x3d) {
        av_log(avctx, AV_LOG_ERROR, "Invalid chunk type %d\n", chunk_type);
        return AVERROR_INVALIDDATA;
    ret = brpix_decode_header(&hdr, &gb);
    if (!ret) {
        av_log(avctx, AV_LOG_ERROR, "Invalid header length\n");
        return AVERROR_INVALIDDATA;
    switch (hdr.format) {
    case 3:
        avctx->pix_fmt = AV_PIX_FMT_PAL8;
        bytes_pp = 1;
        break;
    case 4:
        avctx->pix_fmt = AV_PIX_FMT_RGB555BE;
        bytes_pp = 2;
        break;
    case 5:
        avctx->pix_fmt = AV_PIX_FMT_RGB565BE;
        bytes_pp = 2;
        break;
    case 6:
        avctx->pix_fmt = AV_PIX_FMT_RGB24;
        bytes_pp = 3;
        break;
    case 7:
        avctx->pix_fmt = AV_PIX_FMT_0RGB;
        bytes_pp = 4;
        break;
    case 18:
        avctx->pix_fmt = AV_PIX_FMT_GRAY8A;
        bytes_pp = 2;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Format %d is not supported\n",
                                    hdr.format);
        return AVERROR_PATCHWELCOME;
    if (s->frame.data[0])
        avctx->release_buffer(avctx, &s->frame);
    if (av_image_check_size(hdr.width, hdr.height, 0, avctx) < 0)
        return AVERROR_INVALIDDATA;
    if (hdr.width != avctx->width || hdr.height != avctx->height)
        avcodec_set_dimensions(avctx, hdr.width, hdr.height);
    if ((ret = ff_get_buffer(avctx, &s->frame)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return ret;
    chunk_type = bytestream2_get_be32(&gb);
    if (avctx->pix_fmt == AV_PIX_FMT_PAL8 &&
        (chunk_type == 0x3 || chunk_type == 0x3d)) {
        BRPixHeader palhdr;
        ret = brpix_decode_header(&palhdr, &gb);
        if (!ret) {
            av_log(avctx, AV_LOG_ERROR, "Invalid palette header length\n");
            return AVERROR_INVALIDDATA;
        if (palhdr.format != 7) {
            av_log(avctx, AV_LOG_ERROR, "Palette is not in 0RGB format\n");
            return AVERROR_INVALIDDATA;
        chunk_type = bytestream2_get_be32(&gb);
        data_len = bytestream2_get_be32(&gb);
        bytestream2_skip(&gb, 8);
        if (chunk_type != 0x21 || data_len != 1032 ||
            bytestream2_get_bytes_left(&gb) < 1032) {
            av_log(avctx, AV_LOG_ERROR, "Invalid palette data\n");
            return AVERROR_INVALIDDATA;
        // convert 0RGB to machine endian format (ARGB32)
            bytestream2_skipu(&gb, 1);
            *pal_out++ = (0xFFU << 24) | bytestream2_get_be24u(&gb);
        bytestream2_skip(&gb, 8);
        chunk_type = bytestream2_get_be32(&gb);
    data_len = bytestream2_get_be32(&gb);
    bytestream2_skip(&gb, 8);
    // read the image data to the buffer
    {
        unsigned int bytes_per_scanline = bytes_pp * hdr.width;
        unsigned int bytes_left = bytestream2_get_bytes_left(&gb);
        if (chunk_type != 0x21 || data_len != bytes_left ||
            bytes_left / bytes_per_scanline < hdr.height)
        {
            av_log(avctx, AV_LOG_ERROR, "Invalid image data\n");
            return AVERROR_INVALIDDATA;
        av_image_copy_plane(s->frame.data[0], s->frame.linesize[0],
                            avpkt->data + bytestream2_tell(&gb),
                            bytes_per_scanline,
                            bytes_per_scanline, hdr.height);
    *frame_out = s->frame;
    *got_frame = 1;
    return avpkt->size;