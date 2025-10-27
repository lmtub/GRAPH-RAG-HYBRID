static int opt_preset(const char *opt, const char *arg)

{

    FILE *f=NULL;

    char tmp[1000], tmp2[1000], line[1000];

    int i;

    const char *base[3]= { getenv("HOME"),

                           "/usr/local/share",

                           "/usr/share",

                         };



    for(i=!base[0]; i<3 && !f; i++){

        snprintf(tmp, sizeof(tmp), "%s/%sffmpeg/%s.ffpreset", base[i], i ? "" : ".", arg);

        f= fopen(tmp, "r");

        if(!f){

            char *codec_name= *opt == 'v' ? video_codec_name :

                              *opt == 'a' ? audio_codec_name :

                                            subtitle_codec_name;

              snprintf(tmp, sizeof(tmp), "%s/%sffmpeg/%s-%s.ffpreset", base[i],  i ? "" : ".", codec_name, arg);

            f= fopen(tmp, "r");

        }

    }

    if(!f && ((arg[0]=='.' && arg[1]=='/') || arg[0]=='/')){

        f= fopen(arg, "r");

    }



    if(!f){

        fprintf(stderr, "Preset file not found\n");

        av_exit(1);

    }



    while(!feof(f)){

        int e= fscanf(f, "%999[^\n]\n", line) - 1;

        if(line[0] == '#' && !e)

            continue;

        e|= sscanf(line, "%999[^=]=%999[^\n]\n", tmp, tmp2) - 2;

        if(e){

            fprintf(stderr, "Preset file invalid\n");

            av_exit(1);

        }

        if(!strcmp(tmp, "acodec")){

            opt_audio_codec(tmp2);

        }else if(!strcmp(tmp, "vcodec")){

            opt_video_codec(tmp2);

        }else if(!strcmp(tmp, "scodec")){

            opt_subtitle_codec(tmp2);

        }else

            opt_default(tmp, tmp2);

    }



    fclose(f);



    return 0;

}
