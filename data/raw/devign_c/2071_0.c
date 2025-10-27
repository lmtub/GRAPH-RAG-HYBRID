static void dump_metadata(void *ctx, AVDictionary *m, const char *indent)

{

    if(m && !(m->count == 1 && av_dict_get(m, "language", NULL, 0))){

        AVDictionaryEntry *tag=NULL;



        av_log(ctx, AV_LOG_INFO, "%sMetadata:\n", indent);

        while((tag=av_dict_get(m, "", tag, AV_DICT_IGNORE_SUFFIX))) {

            if(strcmp("language", tag->key)){

                const char *p = tag->value;

                av_log(ctx, AV_LOG_INFO, "%s  %-16s: ", indent, tag->key);

                while(*p) {

                    char tmp[256];

                    size_t len = strcspn(p, "\xd\xa");

                    av_strlcpy(tmp, p, FFMIN(sizeof(tmp), len+1));

                    av_log(ctx, AV_LOG_INFO, "%s", tmp);

                    p += len;

                    if (*p == 0xd) av_log(ctx, AV_LOG_INFO, " ");

                    if (*p == 0xa) av_log(ctx, AV_LOG_INFO, "\n%s  %-16s: ", indent, "");

                    if (*p) p++;

                }

                av_log(ctx, AV_LOG_INFO, "\n");

            }

        }

    }

}
