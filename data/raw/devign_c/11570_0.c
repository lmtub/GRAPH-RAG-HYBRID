int av_opt_set_from_string(void *ctx, const char *opts,

                           const char *const *shorthand,

                           const char *key_val_sep, const char *pairs_sep)

{

    int ret, count = 0;

    const char *dummy_shorthand = NULL;

    char key_buf[68], *value;

    const char *key;



    if (!opts)

        return 0;

    if (!shorthand)

        shorthand = &dummy_shorthand;



    while (*opts) {

        if ((ret = get_key(&opts, key_val_sep, key_buf, sizeof(key_buf))) < 0) {

            if (*shorthand) {

                key = *(shorthand++);

            } else {

                av_log(ctx, AV_LOG_ERROR, "No option name near '%s'\n", opts);

                return AVERROR(EINVAL);

            }

        } else {

            key = key_buf;

            while (*shorthand) /* discard all remaining shorthand */

                shorthand++;

        }



        if (!(value = av_get_token(&opts, pairs_sep)))

            return AVERROR(ENOMEM);

        if (*opts && strchr(pairs_sep, *opts))

            opts++;



        av_log(ctx, AV_LOG_DEBUG, "Setting '%s' to value '%s'\n", key, value);

        if ((ret = av_opt_set(ctx, key, value, 0)) < 0) {

            if (ret == AVERROR_OPTION_NOT_FOUND)

                av_log(ctx, AV_LOG_ERROR, "Option '%s' not found\n", key);

            av_free(value);

            return ret;

        }



        av_free(value);

        count++;

    }

    return count;

}
