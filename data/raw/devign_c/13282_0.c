static int open_input(struct variant *var)

{

    struct segment *seg = var->segments[var->cur_seq_no - var->start_seq_no];

    if (seg->key_type == KEY_NONE) {

        return ffurl_open(&var->input, seg->url, AVIO_FLAG_READ,

                          &var->parent->interrupt_callback, NULL);

    } else if (seg->key_type == KEY_AES_128) {

        char iv[33], key[33], url[MAX_URL_SIZE];

        int ret;

        if (strcmp(seg->key, var->key_url)) {

            URLContext *uc;

            if (ffurl_open(&uc, seg->key, AVIO_FLAG_READ,

                           &var->parent->interrupt_callback, NULL) == 0) {

                if (ffurl_read_complete(uc, var->key, sizeof(var->key))

                    != sizeof(var->key)) {

                    av_log(NULL, AV_LOG_ERROR, "Unable to read key file %s\n",

                           seg->key);

                }

                ffurl_close(uc);

            } else {

                av_log(NULL, AV_LOG_ERROR, "Unable to open key file %s\n",

                       seg->key);

            }

            av_strlcpy(var->key_url, seg->key, sizeof(var->key_url));

        }

        ff_data_to_hex(iv, seg->iv, sizeof(seg->iv), 0);

        ff_data_to_hex(key, var->key, sizeof(var->key), 0);

        iv[32] = key[32] = '\0';

        if (strstr(seg->url, "://"))

            snprintf(url, sizeof(url), "crypto+%s", seg->url);

        else

            snprintf(url, sizeof(url), "crypto:%s", seg->url);

        if ((ret = ffurl_alloc(&var->input, url, AVIO_FLAG_READ,

                               &var->parent->interrupt_callback)) < 0)

            return ret;

        av_opt_set(var->input->priv_data, "key", key, 0);

        av_opt_set(var->input->priv_data, "iv", iv, 0);

        if ((ret = ffurl_connect(var->input, NULL)) < 0) {

            ffurl_close(var->input);

            var->input = NULL;

            return ret;

        }

        return 0;

    }

    return AVERROR(ENOSYS);

}
