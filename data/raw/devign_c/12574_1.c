static int read_decode_block(ALSDecContext *ctx, ALSBlockData *bd)

{

    int ret;



    ret = read_block(ctx, bd);



    if (ret)

        return ret;



    ret = decode_block(ctx, bd);



    return ret;

}
