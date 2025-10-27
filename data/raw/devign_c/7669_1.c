static int dvbsub_parse_region_segment(AVCodecContext *avctx,
                                       const uint8_t *buf, int buf_size)
{
    DVBSubContext *ctx = avctx->priv_data;
    const uint8_t *buf_end = buf + buf_size;
    int region_id, object_id;
    int av_unused version;
    DVBSubRegion *region;
    DVBSubObject *object;
    DVBSubObjectDisplay *display;
    int fill;
    int ret;
    if (buf_size < 10)
        return AVERROR_INVALIDDATA;
    region_id = *buf++;
    region = get_region(ctx, region_id);
    if (!region) {
        region = av_mallocz(sizeof(DVBSubRegion));
        if (!region)
            return AVERROR(ENOMEM);
        region->id = region_id;
        region->version = -1;
        region->next = ctx->region_list;
        ctx->region_list = region;
    version = ((*buf)>>4) & 15;
    fill = ((*buf++) >> 3) & 1;
    region->width = AV_RB16(buf);
    buf += 2;
    region->height = AV_RB16(buf);
    buf += 2;
    if (region->width * region->height != region->buf_size) {
        av_free(region->pbuf);
        region->buf_size = region->width * region->height;
        region->pbuf = av_malloc(region->buf_size);
        if (!region->pbuf) {
            region->buf_size =
            region->width =
            region->height = 0;
            return AVERROR(ENOMEM);
        fill = 1;
        region->dirty = 0;
    region->depth = 1 << (((*buf++) >> 2) & 7);
    if(region->depth<2 || region->depth>8){
        av_log(avctx, AV_LOG_ERROR, "region depth %d is invalid\n", region->depth);
        region->depth= 4;
    region->clut = *buf++;
    if (region->depth == 8) {
        region->bgcolor = *buf++;
        buf += 1;
    } else {
        buf += 1;
        if (region->depth == 4)
            region->bgcolor = (((*buf++) >> 4) & 15);
        else
            region->bgcolor = (((*buf++) >> 2) & 3);
    ff_dlog(avctx, "Region %d, (%dx%d)\n", region_id, region->width, region->height);
    if (fill) {
        memset(region->pbuf, region->bgcolor, region->buf_size);
        ff_dlog(avctx, "Fill region (%d)\n", region->bgcolor);
    delete_region_display_list(ctx, region);
    while (buf + 5 < buf_end) {
        object_id = AV_RB16(buf);
        buf += 2;
        object = get_object(ctx, object_id);
        if (!object) {
            object = av_mallocz(sizeof(DVBSubObject));
            if (!object)
                return AVERROR(ENOMEM);
            object->id = object_id;
            object->next = ctx->object_list;
            ctx->object_list = object;
        object->type = (*buf) >> 6;
        display = av_mallocz(sizeof(DVBSubObjectDisplay));
        if (!display)
            return AVERROR(ENOMEM);
        display->object_id = object_id;
        display->region_id = region_id;
        display->x_pos = AV_RB16(buf) & 0xfff;
        buf += 2;
        display->y_pos = AV_RB16(buf) & 0xfff;
        buf += 2;
        if ((object->type == 1 || object->type == 2) && buf+1 < buf_end) {
            display->fgcolor = *buf++;
            display->bgcolor = *buf++;
        display->region_list_next = region->display_list;
        region->display_list = display;
        display->object_list_next = object->display_list;
        object->display_list = display;
    return 0;