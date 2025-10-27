int attribute_align_arg sws_scale(struct SwsContext *c,
                                  const uint8_t * const srcSlice[],
                                  const int srcStride[], int srcSliceY,
                                  int srcSliceH, uint8_t *const dst[],
                                  const int dstStride[])
{
    int i, ret;
    const uint8_t *src2[4];
    uint8_t *dst2[4];
    uint8_t *rgb0_tmp = NULL;
    if (!srcStride || !dstStride || !dst || !srcSlice) {
        av_log(c, AV_LOG_ERROR, "One of the input parameters to sws_scale() is NULL, please check the calling code\n");
        return 0;
    if (c->gamma_flag && c->cascaded_context[0]) {
        ret = sws_scale(c->cascaded_context[0],
                    srcSlice, srcStride, srcSliceY, srcSliceH,
                    c->cascaded_tmp, c->cascaded_tmpStride);
        if (ret < 0)
            return ret;
        if (c->cascaded_context[2])
            ret = sws_scale(c->cascaded_context[1], (const uint8_t * const *)c->cascaded_tmp, c->cascaded_tmpStride, srcSliceY, srcSliceH, c->cascaded1_tmp, c->cascaded1_tmpStride);
        else
            ret = sws_scale(c->cascaded_context[1], (const uint8_t * const *)c->cascaded_tmp, c->cascaded_tmpStride, srcSliceY, srcSliceH, dst, dstStride);
        if (ret < 0)
            return ret;
        if (c->cascaded_context[2]) {
            ret = sws_scale(c->cascaded_context[2],
                        (const uint8_t * const *)c->cascaded1_tmp, c->cascaded1_tmpStride, c->cascaded_context[1]->dstY - ret, c->cascaded_context[1]->dstY,
                        dst, dstStride);
        return ret;
    if (c->cascaded_context[0] && srcSliceY == 0 && srcSliceH == c->cascaded_context[0]->srcH) {
        ret = sws_scale(c->cascaded_context[0],
                        srcSlice, srcStride, srcSliceY, srcSliceH,
                        c->cascaded_tmp, c->cascaded_tmpStride);
        if (ret < 0)
            return ret;
        ret = sws_scale(c->cascaded_context[1],
                        (const uint8_t * const * )c->cascaded_tmp, c->cascaded_tmpStride, 0, c->cascaded_context[0]->dstH,
                        dst, dstStride);
        return ret;
    memcpy(src2, srcSlice, sizeof(src2));
    memcpy(dst2, dst, sizeof(dst2));
    // do not mess up sliceDir if we have a "trailing" 0-size slice
    if (srcSliceH == 0)
        return 0;
    if (!check_image_pointers(srcSlice, c->srcFormat, srcStride)) {
        av_log(c, AV_LOG_ERROR, "bad src image pointers\n");
        return 0;
    if (!check_image_pointers((const uint8_t* const*)dst, c->dstFormat, dstStride)) {
        av_log(c, AV_LOG_ERROR, "bad dst image pointers\n");
        return 0;
    if (c->sliceDir == 0 && srcSliceY != 0 && srcSliceY + srcSliceH != c->srcH) {
        av_log(c, AV_LOG_ERROR, "Slices start in the middle!\n");
        return 0;
    if (c->sliceDir == 0) {
        if (srcSliceY == 0) c->sliceDir = 1; else c->sliceDir = -1;
    if (usePal(c->srcFormat)) {
        for (i = 0; i < 256; i++) {
            int r, g, b, y, u, v, a = 0xff;
            if (c->srcFormat == AV_PIX_FMT_PAL8) {
                uint32_t p = ((const uint32_t *)(srcSlice[1]))[i];
                a = (p >> 24) & 0xFF;
                r = (p >> 16) & 0xFF;
                g = (p >>  8) & 0xFF;
                b =  p        & 0xFF;
            } else if (c->srcFormat == AV_PIX_FMT_RGB8) {
                r = ( i >> 5     ) * 36;
                g = ((i >> 2) & 7) * 36;
                b = ( i       & 3) * 85;
            } else if (c->srcFormat == AV_PIX_FMT_BGR8) {
                b = ( i >> 6     ) * 85;
                g = ((i >> 3) & 7) * 36;
                r = ( i       & 7) * 36;
            } else if (c->srcFormat == AV_PIX_FMT_RGB4_BYTE) {
                r = ( i >> 3     ) * 255;
                g = ((i >> 1) & 3) * 85;
                b = ( i       & 1) * 255;
            } else if (c->srcFormat == AV_PIX_FMT_GRAY8 || c->srcFormat == AV_PIX_FMT_GRAY8A) {
                r = g = b = i;
            } else {
                av_assert1(c->srcFormat == AV_PIX_FMT_BGR4_BYTE);
                b = ( i >> 3     ) * 255;
                g = ((i >> 1) & 3) * 85;
                r = ( i       & 1) * 255;
#define RGB2YUV_SHIFT 15
#define BY ( (int) (0.114 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define BV (-(int) (0.081 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define BU ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GY ( (int) (0.587 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GV (-(int) (0.419 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GU (-(int) (0.331 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RY ( (int) (0.299 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RV ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RU (-(int) (0.169 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
            y = av_clip_uint8((RY * r + GY * g + BY * b + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
            u = av_clip_uint8((RU * r + GU * g + BU * b + (257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
            v = av_clip_uint8((RV * r + GV * g + BV * b + (257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
            c->pal_yuv[i]= y + (u<<8) + (v<<16) + ((unsigned)a<<24);
            switch (c->dstFormat) {
            case AV_PIX_FMT_BGR32:
#if !HAVE_BIGENDIAN
            case AV_PIX_FMT_RGB24:
#endif
                c->pal_rgb[i]=  r + (g<<8) + (b<<16) + ((unsigned)a<<24);
                break;
            case AV_PIX_FMT_BGR32_1:
#if HAVE_BIGENDIAN
            case AV_PIX_FMT_BGR24:
#endif
                c->pal_rgb[i]= a + (r<<8) + (g<<16) + ((unsigned)b<<24);
                break;
            case AV_PIX_FMT_RGB32_1:
#if HAVE_BIGENDIAN
            case AV_PIX_FMT_RGB24:
#endif
                c->pal_rgb[i]= a + (b<<8) + (g<<16) + ((unsigned)r<<24);
                break;
            case AV_PIX_FMT_RGB32:
#if !HAVE_BIGENDIAN
            case AV_PIX_FMT_BGR24:
#endif
            default:
                c->pal_rgb[i]=  b + (g<<8) + (r<<16) + ((unsigned)a<<24);
    if (c->src0Alpha && !c->dst0Alpha && isALPHA(c->dstFormat)) {
        uint8_t *base;
        int x,y;
        rgb0_tmp = av_malloc(FFABS(srcStride[0]) * srcSliceH + 32);
        if (!rgb0_tmp)
            return AVERROR(ENOMEM);
        base = srcStride[0] < 0 ? rgb0_tmp - srcStride[0] * (srcSliceH-1) : rgb0_tmp;
        for (y=0; y<srcSliceH; y++){
            memcpy(base + srcStride[0]*y, src2[0] + srcStride[0]*y, 4*c->srcW);
            for (x=c->src0Alpha-1; x<4*c->srcW; x+=4) {
                base[ srcStride[0]*y + x] = 0xFF;
        src2[0] = base;
    if (c->srcXYZ && !(c->dstXYZ && c->srcW==c->dstW && c->srcH==c->dstH)) {
        uint8_t *base;
        rgb0_tmp = av_malloc(FFABS(srcStride[0]) * srcSliceH + 32);
        if (!rgb0_tmp)
            return AVERROR(ENOMEM);
        base = srcStride[0] < 0 ? rgb0_tmp - srcStride[0] * (srcSliceH-1) : rgb0_tmp;
        xyz12Torgb48(c, (uint16_t*)base, (const uint16_t*)src2[0], srcStride[0]/2, srcSliceH);
        src2[0] = base;
    if (!srcSliceY && (c->flags & SWS_BITEXACT) && c->dither == SWS_DITHER_ED && c->dither_error[0])
        for (i = 0; i < 4; i++)
            memset(c->dither_error[i], 0, sizeof(c->dither_error[0][0]) * (c->dstW+2));
    // copy strides, so they can safely be modified
    if (c->sliceDir == 1) {
        // slices go from top to bottom
        int srcStride2[4] = { srcStride[0], srcStride[1], srcStride[2],
                              srcStride[3] };
        int dstStride2[4] = { dstStride[0], dstStride[1], dstStride[2],
                              dstStride[3] };
        reset_ptr(src2, c->srcFormat);
        reset_ptr((void*)dst2, c->dstFormat);
        /* reset slice direction at end of frame */
        if (srcSliceY + srcSliceH == c->srcH)
            c->sliceDir = 0;
        ret = c->swscale(c, src2, srcStride2, srcSliceY, srcSliceH, dst2,
                          dstStride2);
    } else {
        // slices go from bottom to top => we flip the image internally
        int srcStride2[4] = { -srcStride[0], -srcStride[1], -srcStride[2],
                              -srcStride[3] };
        int dstStride2[4] = { -dstStride[0], -dstStride[1], -dstStride[2],
                              -dstStride[3] };
        src2[0] += (srcSliceH - 1) * srcStride[0];
        if (!usePal(c->srcFormat))
            src2[1] += ((srcSliceH >> c->chrSrcVSubSample) - 1) * srcStride[1];
        src2[2] += ((srcSliceH >> c->chrSrcVSubSample) - 1) * srcStride[2];
        src2[3] += (srcSliceH - 1) * srcStride[3];
        dst2[0] += ( c->dstH                         - 1) * dstStride[0];
        dst2[1] += ((c->dstH >> c->chrDstVSubSample) - 1) * dstStride[1];
        dst2[2] += ((c->dstH >> c->chrDstVSubSample) - 1) * dstStride[2];
        dst2[3] += ( c->dstH                         - 1) * dstStride[3];
        reset_ptr(src2, c->srcFormat);
        reset_ptr((void*)dst2, c->dstFormat);
        /* reset slice direction at end of frame */
        if (!srcSliceY)
            c->sliceDir = 0;
        ret = c->swscale(c, src2, srcStride2, c->srcH-srcSliceY-srcSliceH,
                          srcSliceH, dst2, dstStride2);
    if (c->dstXYZ && !(c->srcXYZ && c->srcW==c->dstW && c->srcH==c->dstH)) {
        /* replace on the same data */
        rgb48Toxyz12(c, (uint16_t*)dst2[0], (const uint16_t*)dst2[0], dstStride[0]/2, ret);
    av_free(rgb0_tmp);
    return ret;