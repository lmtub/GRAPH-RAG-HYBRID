static void print_report(AVFormatContext **output_files,

                         AVOutputStream **ost_table, int nb_ostreams,

                         int is_last_report)

{

    char buf[1024];

    AVOutputStream *ost;

    AVFormatContext *oc, *os;

    int64_t total_size;

    AVCodecContext *enc;

    int frame_number, vid, i;

    double bitrate, ti1, pts;

    static int64_t last_time = -1;

    static int qp_histogram[52];



    if (!is_last_report) {

        int64_t cur_time;

        /* display the report every 0.5 seconds */

        cur_time = av_gettime();

        if (last_time == -1) {

            last_time = cur_time;

            return;

        }

        if ((cur_time - last_time) < 500000)

            return;

        last_time = cur_time;

    }





    oc = output_files[0];



    total_size = url_fsize(oc->pb);

    if(total_size<0) // FIXME improve url_fsize() so it works with non seekable output too

        total_size= url_ftell(oc->pb);



    buf[0] = '\0';

    ti1 = 1e10;

    vid = 0;

    for(i=0;i<nb_ostreams;i++) {

        ost = ost_table[i];

        os = output_files[ost->file_index];

        enc = ost->st->codec;

        if (vid && enc->codec_type == CODEC_TYPE_VIDEO) {

            snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "q=%2.1f ",

                    enc->coded_frame->quality/(float)FF_QP2LAMBDA);

        }

        if (!vid && enc->codec_type == CODEC_TYPE_VIDEO) {

            float t = (av_gettime()-timer_start) / 1000000.0;



            frame_number = ost->frame_number;

            snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "frame=%5d fps=%3d q=%3.1f ",

                     frame_number, (t>1)?(int)(frame_number/t+0.5) : 0,

                     enc->coded_frame ? enc->coded_frame->quality/(float)FF_QP2LAMBDA : -1);

            if(is_last_report)

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "L");

            if(qp_hist && enc->coded_frame){

                int j;

                int qp= lrintf(enc->coded_frame->quality/(float)FF_QP2LAMBDA);

                if(qp>=0 && qp<sizeof(qp_histogram)/sizeof(int))

                    qp_histogram[qp]++;

                for(j=0; j<32; j++)

                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "%X", (int)lrintf(log(qp_histogram[j]+1)/log(2)));

            }

            if (enc->flags&CODEC_FLAG_PSNR){

                int j;

                double error, error_sum=0;

                double scale, scale_sum=0;

                char type[3]= {'Y','U','V'};

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "PSNR=");

                for(j=0; j<3; j++){

                    if(is_last_report){

                        error= enc->error[j];

                        scale= enc->width*enc->height*255.0*255.0*frame_number;

                    }else{

                        error= enc->coded_frame->error[j];

                        scale= enc->width*enc->height*255.0*255.0;

                    }

                    if(j) scale/=4;

                    error_sum += error;

                    scale_sum += scale;

                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "%c:%2.2f ", type[j], psnr(error/scale));

                }

                snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), "*:%2.2f ", psnr(error_sum/scale_sum));

            }

            vid = 1;

        }

        /* compute min output value */

        pts = (double)ost->st->pts.val * av_q2d(ost->st->time_base);

        if ((pts < ti1) && (pts > 0))

            ti1 = pts;

    }

    if (ti1 < 0.01)

        ti1 = 0.01;



    if (verbose || is_last_report) {

        bitrate = (double)(total_size * 8) / ti1 / 1000.0;



        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),

            "size=%8.0fkB time=%0.1f bitrate=%6.1fkbits/s",

            (double)total_size / 1024, ti1, bitrate);



        if (verbose > 1)

          snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " dup=%d drop=%d",

                  nb_frames_dup, nb_frames_drop);



        if (verbose >= 0)

            fprintf(stderr, "%s    \r", buf);



        fflush(stderr);

    }



    if (is_last_report && verbose >= 0){

        int64_t raw= audio_size + video_size + extra_size;

        fprintf(stderr, "\n");

        fprintf(stderr, "video:%1.0fkB audio:%1.0fkB global headers:%1.0fkB muxing overhead %f%%\n",

                video_size/1024.0,

                audio_size/1024.0,

                extra_size/1024.0,

                100.0*(total_size - raw)/raw

        );

    }

}
