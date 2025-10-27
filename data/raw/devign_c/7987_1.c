static void bastardized_rice_decompress(ALACContext *alac,

                                 int32_t *output_buffer,

                                 int output_size,

                                 int readsamplesize, /* arg_10 */

                                 int rice_initialhistory, /* arg424->b */

                                 int rice_kmodifier, /* arg424->d */

                                 int rice_historymult, /* arg424->c */

                                 int rice_kmodifier_mask /* arg424->e */

        )

{

    int output_count;

    unsigned int history = rice_initialhistory;

    int sign_modifier = 0;



    for (output_count = 0; output_count < output_size; output_count++) {

        int32_t x;

        int32_t x_modified;

        int32_t final_val;



        /* standard rice encoding */

        int k; /* size of extra bits */



        /* read k, that is bits as is */

        k = av_log2((history >> 9) + 3);

        x= decode_scalar(&alac->gb, k, rice_kmodifier, readsamplesize);



        x_modified = sign_modifier + x;

        final_val = (x_modified + 1) / 2;

        if (x_modified & 1) final_val *= -1;



        output_buffer[output_count] = final_val;



        sign_modifier = 0;



        /* now update the history */

        history += x_modified * rice_historymult

                   - ((history * rice_historymult) >> 9);



        if (x_modified > 0xffff)

            history = 0xffff;



        /* special case: there may be compressed blocks of 0 */

        if ((history < 128) && (output_count+1 < output_size)) {

            int k;

            unsigned int block_size;



            sign_modifier = 1;



            k = 7 - av_log2(history) + ((history + 16) >> 6 /* / 64 */);



            block_size= decode_scalar(&alac->gb, k, rice_kmodifier, 16);



            if (block_size > 0) {

                if(block_size >= output_size - output_count){

                    av_log(alac->avctx, AV_LOG_ERROR, "invalid zero block size of %d %d %d\n", block_size, output_size, output_count);

                    block_size= output_size - output_count - 1;

                }

                memset(&output_buffer[output_count+1], 0, block_size * 4);

                output_count += block_size;

            }



            if (block_size > 0xffff)

                sign_modifier = 0;



            history = 0;

        }

    }

}
