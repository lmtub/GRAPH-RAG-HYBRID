static void do_video_stats(AVOutputStream *ost, 

                         AVInputStream *ist,

                         int frame_size)

{

    static FILE *fvstats=NULL;

    static INT64 total_size = 0;

    struct tm *today;

    time_t today2;

    char filename[40];

    AVCodecContext *enc;

    int frame_number;

    INT64 ti;

    double ti1, bitrate, avg_bitrate;

    

    if (!fvstats) {

        today2 = time(NULL);

        today = localtime(&today2);

        sprintf(filename, "vstats_%02d%02d%02d.log", today->tm_hour,

                                               today->tm_min,

                                               today->tm_sec);

        fvstats = fopen(filename,"w");

        if (!fvstats) {

            perror("fopen");

            exit(1);

        }

    }

    

    ti = MAXINT64;

    enc = &ost->st->codec;

    total_size += frame_size;

    if (enc->codec_type == CODEC_TYPE_VIDEO) {

        frame_number = ist->frame_number;

        fprintf(fvstats, "frame= %5d q= %2d ", frame_number, enc->quality);

        if (do_psnr)

            fprintf(fvstats, "PSNR= %6.2f ", enc->psnr_y);

        

        fprintf(fvstats,"f_size= %6d ", frame_size);

        /* compute min pts value */

        if (!ist->discard && ist->pts < ti) {

            ti = ist->pts;

        }

        ti1 = (double)ti / 1000000.0;

        if (ti1 < 0.01)

            ti1 = 0.01;

    

        bitrate = (double)(frame_size * 8) * enc->frame_rate / FRAME_RATE_BASE / 1000.0;

        avg_bitrate = (double)(total_size * 8) / ti1 / 1000.0;

        fprintf(fvstats, "s_size= %8.0fkB time= %0.3f br= %7.1fkbits/s avg_br= %7.1fkbits/s ",

            (double)total_size / 1024, ti1, bitrate, avg_bitrate);

        fprintf(fvstats,"type= %s\n", enc->key_frame == 1 ? "I" : "P");        

    }



    

    

}
