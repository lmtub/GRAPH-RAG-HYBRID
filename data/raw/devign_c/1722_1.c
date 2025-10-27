static int decode_block(ALSDecContext *ctx, ALSBlockData *bd)

{

    unsigned int smp;



    // read block type flag and read the samples accordingly

    if (*bd->const_block)

        decode_const_block_data(ctx, bd);

    else if (decode_var_block_data(ctx, bd))

        return -1;



    // TODO: read RLSLMS extension data



    if (*bd->shift_lsbs)

        for (smp = 0; smp < bd->block_length; smp++)

            bd->raw_samples[smp] <<= *bd->shift_lsbs;



    return 0;

}
