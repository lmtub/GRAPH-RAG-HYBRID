static int decode_frame_header(ProresContext *ctx, const uint8_t *buf,

                               const int data_size, AVCodecContext *avctx)

{

    int hdr_size, width, height, flags;

    int version;

    const uint8_t *ptr;



    hdr_size = AV_RB16(buf);

    av_dlog(avctx, "header size %d\n", hdr_size);

    if (hdr_size > data_size) {

        av_log(avctx, AV_LOG_ERROR, "error, wrong header size\n");





    version = AV_RB16(buf + 2);

    av_dlog(avctx, "%.4s version %d\n", buf+4, version);

    if (version > 1) {

        av_log(avctx, AV_LOG_ERROR, "unsupported version: %d\n", version);





    width  = AV_RB16(buf + 8);

    height = AV_RB16(buf + 10);

    if (width != avctx->width || height != avctx->height) {

        av_log(avctx, AV_LOG_ERROR, "picture resolution change: %dx%d -> %dx%d\n",

               avctx->width, avctx->height, width, height);





    ctx->frame_type = (buf[12] >> 2) & 3;



    av_dlog(avctx, "frame type %d\n", ctx->frame_type);



    if (ctx->frame_type == 0) {

        ctx->scan = ctx->progressive_scan; // permuted

    } else {

        ctx->scan = ctx->interlaced_scan; // permuted

        ctx->frame.interlaced_frame = 1;

        ctx->frame.top_field_first = ctx->frame_type == 1;




    avctx->pix_fmt = (buf[12] & 0xC0) == 0xC0 ? AV_PIX_FMT_YUV444P10 : AV_PIX_FMT_YUV422P10;



    ptr   = buf + 20;

    flags = buf[19];

    av_dlog(avctx, "flags %x\n", flags);



    if (flags & 2) {





        permute(ctx->qmat_luma, ctx->prodsp.idct_permutation, ptr);

        ptr += 64;

    } else {

        memset(ctx->qmat_luma, 4, 64);




    if (flags & 1) {





        permute(ctx->qmat_chroma, ctx->prodsp.idct_permutation, ptr);

    } else {

        memset(ctx->qmat_chroma, 4, 64);




    return hdr_size;
