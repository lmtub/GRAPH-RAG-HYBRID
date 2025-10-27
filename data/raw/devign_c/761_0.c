void set_context_opts(void *ctx, void *opts_ctx, int flags, AVCodec *codec)

{

    int i;

    void *priv_ctx=NULL;

    if(!strcmp("AVCodecContext", (*(AVClass**)ctx)->class_name)){

        AVCodecContext *avctx= ctx;

        if(codec && codec->priv_class && avctx->priv_data){

            priv_ctx= avctx->priv_data;

        }

    } else if (!strcmp("AVFormatContext", (*(AVClass**)ctx)->class_name)) {

        AVFormatContext *avctx = ctx;

        if (avctx->oformat && avctx->oformat->priv_class) {

            priv_ctx = avctx->priv_data;

        }

    }



    for(i=0; i<opt_name_count; i++){

        char buf[256];

        const AVOption *opt;

        const char *str= av_get_string(opts_ctx, opt_names[i], &opt, buf, sizeof(buf));

        /* if an option with name opt_names[i] is present in opts_ctx then str is non-NULL */

        if(str && ((opt->flags & flags) == flags))

            av_set_string3(ctx, opt_names[i], str, 1, NULL);

        /* We need to use a differnt system to pass options to the private context because

           it is not known which codec and thus context kind that will be when parsing options

           we thus use opt_values directly instead of opts_ctx */

        if(!str && priv_ctx) {

            if (av_find_opt(priv_ctx, opt_names[i], NULL, flags, flags))

                av_set_string3(priv_ctx, opt_names[i], opt_values[i], 0, NULL);

        }

    }

}
