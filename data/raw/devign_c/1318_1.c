static int get_siz(Jpeg2000DecoderContext *s)
{
    int i;
    int ncomponents;
    uint32_t log2_chroma_wh = 0;
    const enum AVPixelFormat *possible_fmts = NULL;
    int possible_fmts_nb = 0;
    if (bytestream2_get_bytes_left(&s->g) < 36) {
        av_log(s->avctx, AV_LOG_ERROR, "Insufficient space for SIZ\n");
    s->avctx->profile = bytestream2_get_be16u(&s->g); // Rsiz
    s->width          = bytestream2_get_be32u(&s->g); // Width
    s->height         = bytestream2_get_be32u(&s->g); // Height
    s->image_offset_x = bytestream2_get_be32u(&s->g); // X0Siz
    s->image_offset_y = bytestream2_get_be32u(&s->g); // Y0Siz
    s->tile_width     = bytestream2_get_be32u(&s->g); // XTSiz
    s->tile_height    = bytestream2_get_be32u(&s->g); // YTSiz
    s->tile_offset_x  = bytestream2_get_be32u(&s->g); // XT0Siz
    s->tile_offset_y  = bytestream2_get_be32u(&s->g); // YT0Siz
    ncomponents       = bytestream2_get_be16u(&s->g); // CSiz
    if (s->image_offset_x || s->image_offset_y) {
        avpriv_request_sample(s->avctx, "Support for image offsets");
        return AVERROR_PATCHWELCOME;
    if (av_image_check_size(s->width, s->height, 0, s->avctx)) {
        avpriv_request_sample(s->avctx, "Large Dimensions");
        return AVERROR_PATCHWELCOME;
    if (ncomponents <= 0) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid number of components: %d\n",
    if (ncomponents > 4) {
        avpriv_request_sample(s->avctx, "Support for %d components",
                              ncomponents);
        return AVERROR_PATCHWELCOME;
    s->ncomponents = ncomponents;
    if (s->tile_width <= 0 || s->tile_height <= 0) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid tile dimension %dx%d.\n",
               s->tile_width, s->tile_height);
    if (bytestream2_get_bytes_left(&s->g) < 3 * s->ncomponents) {
        av_log(s->avctx, AV_LOG_ERROR, "Insufficient space for %d components in SIZ\n", s->ncomponents);
    for (i = 0; i < s->ncomponents; i++) { // Ssiz_i XRsiz_i, YRsiz_i
        uint8_t x    = bytestream2_get_byteu(&s->g);
        s->cbps[i]   = (x & 0x7f) + 1;
        s->precision = FFMAX(s->cbps[i], s->precision);
        s->sgnd[i]   = !!(x & 0x80);
        s->cdx[i]    = bytestream2_get_byteu(&s->g);
        s->cdy[i]    = bytestream2_get_byteu(&s->g);
        if (   !s->cdx[i] || s->cdx[i] == 3 || s->cdx[i] > 4
            || !s->cdy[i] || s->cdy[i] == 3 || s->cdy[i] > 4) {
            av_log(s->avctx, AV_LOG_ERROR, "Invalid sample separation %d/%d\n", s->cdx[i], s->cdy[i]);
        log2_chroma_wh |= s->cdy[i] >> 1 << i * 4 | s->cdx[i] >> 1 << i * 4 + 2;
    s->numXtiles = ff_jpeg2000_ceildiv(s->width  - s->tile_offset_x, s->tile_width);
    s->numYtiles = ff_jpeg2000_ceildiv(s->height - s->tile_offset_y, s->tile_height);
    if (s->numXtiles * (uint64_t)s->numYtiles > INT_MAX/sizeof(*s->tile)) {
        s->numXtiles = s->numYtiles = 0;
        return AVERROR(EINVAL);
    s->tile = av_mallocz_array(s->numXtiles * s->numYtiles, sizeof(*s->tile));
    if (!s->tile) {
        s->numXtiles = s->numYtiles = 0;
        return AVERROR(ENOMEM);
    for (i = 0; i < s->numXtiles * s->numYtiles; i++) {
        Jpeg2000Tile *tile = s->tile + i;
        tile->comp = av_mallocz(s->ncomponents * sizeof(*tile->comp));
        if (!tile->comp)
            return AVERROR(ENOMEM);
    /* compute image size with reduction factor */
    s->avctx->width  = ff_jpeg2000_ceildivpow2(s->width  - s->image_offset_x,
                                               s->reduction_factor);
    s->avctx->height = ff_jpeg2000_ceildivpow2(s->height - s->image_offset_y,
                                               s->reduction_factor);
    if (s->avctx->profile == FF_PROFILE_JPEG2000_DCINEMA_2K ||
        s->avctx->profile == FF_PROFILE_JPEG2000_DCINEMA_4K) {
        possible_fmts = xyz_pix_fmts;
        possible_fmts_nb = FF_ARRAY_ELEMS(xyz_pix_fmts);
    } else {
        switch (s->colour_space) {
        case 16:
            possible_fmts = rgb_pix_fmts;
            possible_fmts_nb = FF_ARRAY_ELEMS(rgb_pix_fmts);
            break;
        case 17:
            possible_fmts = gray_pix_fmts;
            possible_fmts_nb = FF_ARRAY_ELEMS(gray_pix_fmts);
            break;
        case 18:
            possible_fmts = yuv_pix_fmts;
            possible_fmts_nb = FF_ARRAY_ELEMS(yuv_pix_fmts);
            break;
        default:
            possible_fmts = all_pix_fmts;
            possible_fmts_nb = FF_ARRAY_ELEMS(all_pix_fmts);
            break;
    for (i = 0; i < possible_fmts_nb; ++i) {
        if (pix_fmt_match(possible_fmts[i], ncomponents, s->precision, log2_chroma_wh, s->pal8)) {
            s->avctx->pix_fmt = possible_fmts[i];
            break;
    if (i == possible_fmts_nb) {
        if (ncomponents == 4 &&
            s->cdy[0] == 1 && s->cdx[0] == 1 &&
            s->cdy[1] == 1 && s->cdx[1] == 1 &&
            s->cdy[2] == s->cdy[3] && s->cdx[2] == s->cdx[3]) {
            if (s->precision == 8 && s->cdy[2] == 2 && s->cdx[2] == 2 && !s->pal8) {
                s->avctx->pix_fmt = AV_PIX_FMT_YUVA420P;
                s->cdef[0] = 0;
                s->cdef[1] = 1;
                s->cdef[2] = 2;
                s->cdef[3] = 3;
                i = 0;
    if (i == possible_fmts_nb) {
        av_log(s->avctx, AV_LOG_ERROR,
               "Unknown pix_fmt, profile: %d, colour_space: %d, "
               "components: %d, precision: %d\n"
               "cdx[0]: %d, cdy[0]: %d\n"
               "cdx[1]: %d, cdy[1]: %d\n"
               "cdx[2]: %d, cdy[2]: %d\n"
               "cdx[3]: %d, cdy[3]: %d\n",
               s->avctx->profile, s->colour_space, ncomponents, s->precision,
               s->cdx[0],
               s->cdy[0],
               ncomponents > 1 ? s->cdx[1] : 0,
               ncomponents > 1 ? s->cdy[1] : 0,
               ncomponents > 2 ? s->cdx[2] : 0,
               ncomponents > 2 ? s->cdy[2] : 0,
               ncomponents > 3 ? s->cdx[3] : 0,
               ncomponents > 3 ? s->cdy[3] : 0);
        return AVERROR_PATCHWELCOME;
    s->avctx->bits_per_raw_sample = s->precision;
    return 0;