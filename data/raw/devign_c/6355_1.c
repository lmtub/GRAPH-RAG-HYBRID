static int decode_user_data(MpegEncContext *s, GetBitContext *gb){

    char buf[256];

    int i;

    int e;

    int ver, build, ver2, ver3;

    char last;



    for(i=0; i<255; i++){

        if(show_bits(gb, 23) == 0) break;

        buf[i]= get_bits(gb, 8);

    }

    buf[i]=0;



    /* divx detection */

    e=sscanf(buf, "DivX%dBuild%d%c", &ver, &build, &last);

    if(e<2)

        e=sscanf(buf, "DivX%db%d%c", &ver, &build, &last);

    if(e>=2){

        s->divx_version= ver;

        s->divx_build= build;

        s->divx_packed= e==3 && last=='p';

    }

    

    /* ffmpeg detection */

    e=sscanf(buf, "FFmpe%*[^b]b%d", &build)+3;

    if(e!=4)

        e=sscanf(buf, "FFmpeg v%d.%d.%d / libavcodec build: %d", &ver, &ver2, &ver3, &build); 

    if(e!=4){

        e=sscanf(buf, "Lavc%d.%d.%d", &ver, &ver2, &ver3)+1;

        build= (ver<<16) + (ver2<<8) + ver3;

    }

    if(e!=4){

        if(strcmp(buf, "ffmpeg")==0){

            s->lavc_build= 4600;

        }

    }

    if(e==4){

        s->lavc_build= build;

    }

    

    /* xvid detection */

    e=sscanf(buf, "XviD%d", &build);

    if(e==1){

        s->xvid_build= build;

    }



//printf("User Data: %s\n", buf);

    return 0;

}
