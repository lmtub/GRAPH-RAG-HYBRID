static int parse_object_segment(AVCodecContext *avctx,
                                  const uint8_t *buf, int buf_size)
{
    PGSSubContext *ctx = avctx->priv_data;
    PGSSubObject *object;
    uint8_t sequence_desc;
    unsigned int rle_bitmap_len, width, height;
    int id;
    if (buf_size <= 4)
    buf_size -= 4;
    id = bytestream_get_be16(&buf);
    object = find_object(id, &ctx->objects);
    if (!object) {
        if (ctx->objects.count >= MAX_EPOCH_OBJECTS) {
            av_log(avctx, AV_LOG_ERROR, "Too many objects in epoch\n");
        object = &ctx->objects.object[ctx->objects.count++];
        object->id = id;
    /* skip object version number */
    buf += 1;
    /* Read the Sequence Description to determine if start of RLE data or appended to previous RLE */
    sequence_desc = bytestream_get_byte(&buf);
    if (!(sequence_desc & 0x80)) {
        /* Additional RLE data */
        if (buf_size > object->rle_remaining_len)
        memcpy(object->rle + object->rle_data_len, buf, buf_size);
        object->rle_data_len += buf_size;
        object->rle_remaining_len -= buf_size;
        return 0;
    if (buf_size <= 7)
    buf_size -= 7;
    /* Decode rle bitmap length, stored size includes width/height data */
    rle_bitmap_len = bytestream_get_be24(&buf) - 2*2;
    /* Get bitmap dimensions from data */
    width  = bytestream_get_be16(&buf);
    height = bytestream_get_be16(&buf);
    /* Make sure the bitmap is not too large */
    if (avctx->width < width || avctx->height < height) {
        av_log(avctx, AV_LOG_ERROR, "Bitmap dimensions larger than video.\n");
    object->w = width;
    object->h = height;
    av_fast_malloc(&object->rle, &object->rle_buffer_size, rle_bitmap_len);
    if (!object->rle)
        return AVERROR(ENOMEM);
    memcpy(object->rle, buf, buf_size);
    object->rle_data_len = buf_size;
    object->rle_remaining_len = rle_bitmap_len - buf_size;
    return 0;