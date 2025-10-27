static int opt_list(void *obj, void *av_log_obj, char *unit)

{

    AVOption *opt=NULL;



    while((opt= av_next_option(obj, opt))){

        if(!(opt->flags & (AV_OPT_FLAG_ENCODING_PARAM|AV_OPT_FLAG_DECODING_PARAM)))

            continue;



        /* Don't print CONST's on level one.

         * Don't print anything but CONST's on level two.

         * Only print items from the requested unit.

         */

        if (!unit && opt->type==FF_OPT_TYPE_CONST)

            continue;

        else if (unit && opt->type!=FF_OPT_TYPE_CONST)

            continue;

        else if (unit && opt->type==FF_OPT_TYPE_CONST && strcmp(unit, opt->unit))

            continue;

        else if (unit && opt->type == FF_OPT_TYPE_CONST)

            av_log(av_log_obj, AV_LOG_INFO, "   %-15s ", opt->name);

        else

            av_log(av_log_obj, AV_LOG_INFO, "-%-17s ", opt->name);



        switch( opt->type )

        {

            case FF_OPT_TYPE_FLAGS:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<flags>" );

                break;

            case FF_OPT_TYPE_INT:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<int>" );

                break;

            case FF_OPT_TYPE_INT64:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<int64>" );

                break;

            case FF_OPT_TYPE_DOUBLE:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<double>" );

                break;

            case FF_OPT_TYPE_FLOAT:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<float>" );

                break;

            case FF_OPT_TYPE_STRING:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<string>" );

                break;

            case FF_OPT_TYPE_RATIONAL:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "<rational>" );

                break;

            case FF_OPT_TYPE_CONST:

            default:

                av_log( av_log_obj, AV_LOG_INFO, "%-7s ", "" );

                break;

        }

        av_log(av_log_obj, AV_LOG_INFO, "%c", (opt->flags & AV_OPT_FLAG_ENCODING_PARAM) ? 'E' : '.');

        av_log(av_log_obj, AV_LOG_INFO, "%c", (opt->flags & AV_OPT_FLAG_DECODING_PARAM) ? 'D' : '.');

        av_log(av_log_obj, AV_LOG_INFO, "%c", (opt->flags & AV_OPT_FLAG_VIDEO_PARAM   ) ? 'V' : '.');

        av_log(av_log_obj, AV_LOG_INFO, "%c", (opt->flags & AV_OPT_FLAG_AUDIO_PARAM   ) ? 'A' : '.');

        av_log(av_log_obj, AV_LOG_INFO, "%c", (opt->flags & AV_OPT_FLAG_SUBTITLE_PARAM) ? 'S' : '.');



        if(opt->help)

            av_log(av_log_obj, AV_LOG_INFO, " %s", opt->help);

        av_log(av_log_obj, AV_LOG_INFO, "\n");

        if (opt->unit && opt->type != FF_OPT_TYPE_CONST) {

            opt_list(obj, av_log_obj, opt->unit);

        }

    }

}
