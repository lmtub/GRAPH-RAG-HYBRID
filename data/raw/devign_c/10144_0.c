int avfilter_init_filter(AVFilterContext *filter, const char *args, void *opaque)

#endif

{

    AVDictionary *options = NULL;

    AVDictionaryEntry *e;

    int ret=0;



    if (args && *args) {

        if (!filter->filter->priv_class) {

            av_log(filter, AV_LOG_ERROR, "This filter does not take any "

                   "options, but options were provided: %s.\n", args);

            return AVERROR(EINVAL);

        }



#if FF_API_OLD_FILTER_OPTS

        if (!strcmp(filter->filter->name, "scale") &&

            strchr(args, ':') < strchr(args, '=')) {

            /* old w:h:flags=<flags> syntax */

            char *copy = av_strdup(args);

            char *p;



            av_log(filter, AV_LOG_WARNING, "The <w>:<h>:flags=<flags> option "

                   "syntax is deprecated. Use either <w>:<h>:<flags> or "

                   "w=<w>:h=<h>:flags=<flags>.\n");



            if (!copy) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            p = strrchr(copy, ':');

            if (p) {

                *p++ = 0;

                ret = av_dict_parse_string(&options, p, "=", ":", 0);

            }

            if (ret >= 0)

                ret = process_options(filter, &options, copy);

            av_freep(&copy);



            if (ret < 0)

                goto fail;

        } else if (!strcmp(filter->filter->name, "format")     ||

                   !strcmp(filter->filter->name, "noformat")   ||

                   !strcmp(filter->filter->name, "frei0r")     ||

                   !strcmp(filter->filter->name, "frei0r_src") ||

                   !strcmp(filter->filter->name, "ocv")        ||

                   !strcmp(filter->filter->name, "pan")        ||

                   !strcmp(filter->filter->name, "pp")         ||

                   !strcmp(filter->filter->name, "aevalsrc")) {

            /* a hack for compatibility with the old syntax

             * replace colons with |s */

            char *copy = av_strdup(args);

            char *p    = copy;

            int nb_leading = 0; // number of leading colons to skip

            int deprecated = 0;



            if (!copy) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            if (!strcmp(filter->filter->name, "frei0r") ||

                !strcmp(filter->filter->name, "ocv"))

                nb_leading = 1;

            else if (!strcmp(filter->filter->name, "frei0r_src"))

                nb_leading = 3;



            while (nb_leading--) {

                p = strchr(p, ':');

                if (!p) {

                    p = copy + strlen(copy);

                    break;

                }

                p++;

            }



            deprecated = strchr(p, ':') != NULL;



            if (!strcmp(filter->filter->name, "aevalsrc")) {

                deprecated = 0;

                while ((p = strchr(p, ':')) && p[1] != ':') {

                    const char *epos = strchr(p + 1, '=');

                    const char *spos = strchr(p + 1, ':');

                    const int next_token_is_opt = epos && (!spos || epos < spos);

                    if (next_token_is_opt) {

                        p++;

                        break;

                    }

                    /* next token does not contain a '=', assume a channel expression */

                    deprecated = 1;

                    *p++ = '|';

                }

                if (p && *p == ':') { // double sep '::' found

                    deprecated = 1;

                    memmove(p, p + 1, strlen(p));

                }

            } else

            while ((p = strchr(p, ':')))

                *p++ = '|';



            if (deprecated)

                av_log(filter, AV_LOG_WARNING, "This syntax is deprecated. Use "

                       "'|' to separate the list items.\n");



            av_log(filter, AV_LOG_DEBUG, "compat: called with args=[%s]\n", copy);

            ret = process_options(filter, &options, copy);

            av_freep(&copy);



            if (ret < 0)

                goto fail;

#endif

        } else {

#if CONFIG_MP_FILTER

            if (!strcmp(filter->filter->name, "mp")) {

                char *escaped;



                if (!strncmp(args, "filter=", 7))

                    args += 7;

                ret = av_escape(&escaped, args, ":=", AV_ESCAPE_MODE_BACKSLASH, 0);

                if (ret < 0) {

                    av_log(filter, AV_LOG_ERROR, "Unable to escape MPlayer filters arg '%s'\n", args);

                    goto fail;

                }

                ret = process_options(filter, &options, escaped);

                av_free(escaped);

            } else

#endif

            ret = process_options(filter, &options, args);

            if (ret < 0)

                goto fail;

        }

    }



    if (filter->filter->priv_class) {

        ret = av_opt_set_dict(filter->priv, &options);

        if (ret < 0) {

            av_log(filter, AV_LOG_ERROR, "Error applying options to the filter.\n");

            goto fail;

        }

    }



    if (filter->filter->init_opaque)

        ret = filter->filter->init_opaque(filter, opaque);

    else if (filter->filter->init)

        ret = filter->filter->init(filter);

    else if (filter->filter->init_dict)

        ret = filter->filter->init_dict(filter, &options);

    if (ret < 0)

        goto fail;



    if ((e = av_dict_get(options, "", NULL, AV_DICT_IGNORE_SUFFIX))) {

        av_log(filter, AV_LOG_ERROR, "No such option: %s.\n", e->key);

        ret = AVERROR_OPTION_NOT_FOUND;

        goto fail;

    }



fail:

    av_dict_free(&options);



    return ret;

}
