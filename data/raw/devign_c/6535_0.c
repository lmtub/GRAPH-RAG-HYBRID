static int opt_preset(const char *opt, const char *arg)

{

    FILE *f=NULL;

    char filename[1000], tmp[1000], tmp2[1000], line[1000];

    int i;

    const char *base[3]= { getenv("HOME"),

                           "/usr/local/share",

                           "/usr/share",

                         };



    for(i=!base[0]; i<3 && !f; i++){

        snprintf(filename, sizeof(filename), "%s/%sffmpeg/%s.ffpreset", base[i], i ? "" : ".", arg);

        f= fopen(filename, "r");

        if(!f){

            char *codec_name= *opt == 'v' ? video_codec_name :

                              *opt == 'a' ? audio_codec_name :

                                            subtitle_codec_name;

            snprintf(filename, sizeof(filename), "%s/%sffmpeg/%s-%s.ffpreset", base[i],  i ? "" : ".", codec_name, arg);

            f= fopen(filename, "r");

        }

    }

    if(!f && ((arg[0]=='.' && arg[1]=='/') || arg[0]=='/' ||

              is_dos_path(arg))){

        snprintf(filename, sizeof(filename), arg);

        f= fopen(filename, "r");

    }



    if(!f){

        fprintf(stderr, "File for preset '%s' not found\n", arg);

        av_exit(1);

    }



    while(!feof(f)){

        int e= fscanf(f, "%999[^\n]\n", line) - 1;

        if(line[0] == '#' && !e)

            continue;

        e|= sscanf(line, "%999[^=]=%999[^\n]\n", tmp, tmp2) - 2;

        if(e){

            fprintf(stderr, "%s: Preset file invalid\n", filename);

            av_exit(1);

        }

        if(!strcmp(tmp, "acodec")){

            opt_audio_codec(tmp2);

        }else if(!strcmp(tmp, "vcodec")){

            opt_video_codec(tmp2);

        }else if(!strcmp(tmp, "scodec")){

            opt_subtitle_codec(tmp2);

        }else if(opt_default(tmp, tmp2) < 0){

            fprintf(stderr, "%s: Invalid option or argument: %s=%s\n", filename, tmp, tmp2);

            av_exit(1);

        }

    }



    fclose(f);



    return 0;

}
