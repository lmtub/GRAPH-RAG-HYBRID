static int ffserver_opt_preset(const char *arg,

                       AVCodecContext *avctx, int type,

                       enum AVCodecID *audio_id, enum AVCodecID *video_id)

{

    FILE *f=NULL;

    char filename[1000], tmp[1000], tmp2[1000], line[1000];

    int ret = 0;

    AVCodec *codec = avcodec_find_encoder(avctx->codec_id);



    if (!(f = get_preset_file(filename, sizeof(filename), arg, 0,

                              codec ? codec->name : NULL))) {

        fprintf(stderr, "File for preset '%s' not found\n", arg);

        return 1;

    }



    while(!feof(f)){

        int e= fscanf(f, "%999[^\n]\n", line) - 1;

        if(line[0] == '#' && !e)

            continue;

        e|= sscanf(line, "%999[^=]=%999[^\n]\n", tmp, tmp2) - 2;

        if(e){

            fprintf(stderr, "%s: Invalid syntax: '%s'\n", filename, line);

            ret = 1;

            break;

        }

        if(!strcmp(tmp, "acodec")){

            *audio_id = opt_codec(tmp2, AVMEDIA_TYPE_AUDIO);

        }else if(!strcmp(tmp, "vcodec")){

            *video_id = opt_codec(tmp2, AVMEDIA_TYPE_VIDEO);

        }else if(!strcmp(tmp, "scodec")){

            /* opt_subtitle_codec(tmp2); */

        }else if(ffserver_opt_default(tmp, tmp2, avctx, type) < 0){

            fprintf(stderr, "%s: Invalid option or argument: '%s', parsed as '%s' = '%s'\n", filename, line, tmp, tmp2);

            ret = 1;

            break;

        }

    }



    fclose(f);



    return ret;

}
