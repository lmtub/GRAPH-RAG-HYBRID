static void opt_target(const char *arg)

{

    enum { PAL, NTSC, FILM, UNKNOWN } norm = UNKNOWN;

    static const char *const frame_rates[] = {"25", "30000/1001", "24000/1001"};



    if(!strncmp(arg, "pal-", 4)) {

        norm = PAL;

        arg += 4;

    } else if(!strncmp(arg, "ntsc-", 5)) {

        norm = NTSC;

        arg += 5;

    } else if(!strncmp(arg, "film-", 5)) {

        norm = FILM;

        arg += 5;

    } else {

        int fr;

        /* Calculate FR via float to avoid int overflow */

        fr = (int)(frame_rate.num * 1000.0 / frame_rate.den);

        if(fr == 25000) {

            norm = PAL;

        } else if((fr == 29970) || (fr == 23976)) {

            norm = NTSC;

        } else {

            /* Try to determine PAL/NTSC by peeking in the input files */

            if(nb_input_files) {

                int i, j;

                for(j = 0; j < nb_input_files; j++) {

                    for(i = 0; i < input_files[j]->nb_streams; i++) {

                        AVCodecContext *c = input_files[j]->streams[i]->codec;

                        if(c->codec_type != AVMEDIA_TYPE_VIDEO)

                            continue;

                        fr = c->time_base.den * 1000 / c->time_base.num;

                        if(fr == 25000) {

                            norm = PAL;

                            break;

                        } else if((fr == 29970) || (fr == 23976)) {

                            norm = NTSC;

                            break;

                        }

                    }

                    if(norm != UNKNOWN)

                        break;

                }

            }

        }

        if(verbose && norm != UNKNOWN)

            fprintf(stderr, "Assuming %s for target.\n", norm == PAL ? "PAL" : "NTSC");

    }



    if(norm == UNKNOWN) {

        fprintf(stderr, "Could not determine norm (PAL/NTSC/NTSC-Film) for target.\n");

        fprintf(stderr, "Please prefix target with \"pal-\", \"ntsc-\" or \"film-\",\n");

        fprintf(stderr, "or set a framerate with \"-r xxx\".\n");

        ffmpeg_exit(1);

    }



    if(!strcmp(arg, "vcd")) {



        opt_video_codec("mpeg1video");

        opt_audio_codec("mp2");

        opt_format("vcd");



        opt_frame_size(norm == PAL ? "352x288" : "352x240");

        opt_frame_rate(NULL, frame_rates[norm]);

        opt_default("g", norm == PAL ? "15" : "18");



        opt_default("b", "1150000");

        opt_default("maxrate", "1150000");

        opt_default("minrate", "1150000");

        opt_default("bufsize", "327680"); // 40*1024*8;



        opt_default("ab", "224000");

        audio_sample_rate = 44100;

        audio_channels = 2;



        opt_default("packetsize", "2324");

        opt_default("muxrate", "1411200"); // 2352 * 75 * 8;



        /* We have to offset the PTS, so that it is consistent with the SCR.

           SCR starts at 36000, but the first two packs contain only padding

           and the first pack from the other stream, respectively, may also have

           been written before.

           So the real data starts at SCR 36000+3*1200. */

        mux_preload= (36000+3*1200) / 90000.0; //0.44

    } else if(!strcmp(arg, "svcd")) {



        opt_video_codec("mpeg2video");

        opt_audio_codec("mp2");

        opt_format("svcd");



        opt_frame_size(norm == PAL ? "480x576" : "480x480");

        opt_frame_rate(NULL, frame_rates[norm]);

        opt_default("g", norm == PAL ? "15" : "18");



        opt_default("b", "2040000");

        opt_default("maxrate", "2516000");

        opt_default("minrate", "0"); //1145000;

        opt_default("bufsize", "1835008"); //224*1024*8;

        opt_default("flags", "+scan_offset");





        opt_default("ab", "224000");

        audio_sample_rate = 44100;



        opt_default("packetsize", "2324");



    } else if(!strcmp(arg, "dvd")) {



        opt_video_codec("mpeg2video");

        opt_audio_codec("ac3");

        opt_format("dvd");



        opt_frame_size(norm == PAL ? "720x576" : "720x480");

        opt_frame_rate(NULL, frame_rates[norm]);

        opt_default("g", norm == PAL ? "15" : "18");



        opt_default("b", "6000000");

        opt_default("maxrate", "9000000");

        opt_default("minrate", "0"); //1500000;

        opt_default("bufsize", "1835008"); //224*1024*8;



        opt_default("packetsize", "2048");  // from www.mpucoder.com: DVD sectors contain 2048 bytes of data, this is also the size of one pack.

        opt_default("muxrate", "10080000"); // from mplex project: data_rate = 1260000. mux_rate = data_rate * 8



        opt_default("ab", "448000");

        audio_sample_rate = 48000;



    } else if(!strncmp(arg, "dv", 2)) {



        opt_format("dv");



        opt_frame_size(norm == PAL ? "720x576" : "720x480");

        opt_frame_pix_fmt(!strncmp(arg, "dv50", 4) ? "yuv422p" :

                          (norm == PAL ? "yuv420p" : "yuv411p"));

        opt_frame_rate(NULL, frame_rates[norm]);



        audio_sample_rate = 48000;

        audio_channels = 2;



    } else {

        fprintf(stderr, "Unknown target: %s\n", arg);

        ffmpeg_exit(1);

    }

}
