static void init_entropy_decoder(APEContext *ctx)

{

    /* Read the CRC */

    ctx->CRC = bytestream_get_be32(&ctx->ptr);



    /* Read the frame flags if they exist */

    ctx->frameflags = 0;

    if ((ctx->fileversion > 3820) && (ctx->CRC & 0x80000000)) {

        ctx->CRC &= ~0x80000000;



        ctx->frameflags = bytestream_get_be32(&ctx->ptr);

    }



    /* Keep a count of the blocks decoded in this frame */

    ctx->blocksdecoded = 0;



    /* Initialize the rice structs */

    ctx->riceX.k = 10;

    ctx->riceX.ksum = (1 << ctx->riceX.k) * 16;

    ctx->riceY.k = 10;

    ctx->riceY.ksum = (1 << ctx->riceY.k) * 16;



    /* The first 8 bits of input are ignored. */

    ctx->ptr++;



    range_start_decoding(ctx);

}
