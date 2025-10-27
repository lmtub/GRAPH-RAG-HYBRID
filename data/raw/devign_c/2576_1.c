int av_set_options_string(void *ctx, const char *opts,

                          const char *key_val_sep, const char *pairs_sep)

{

    int ret, count = 0;





    while (*opts) {

        if ((ret = parse_key_value_pair(ctx, &opts, key_val_sep, pairs_sep)) < 0)

            return ret;

        count++;



        if (*opts)

            opts++;

    }



    return count;

}