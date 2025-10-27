int ff_xvid_rate_control_init(MpegEncContext *s){

    char *tmp_name;

    int fd, i;

    xvid_plg_create_t xvid_plg_create;

    xvid_plugin_2pass2_t xvid_2pass2;



//xvid_debug=-1;



    fd=av_tempfile("xvidrc.", &tmp_name, 0, s->avctx);

    if (fd == -1) {

        av_log(NULL, AV_LOG_ERROR, "Can't create temporary pass2 file.\n");

        return -1;

    }



    for(i=0; i<s->rc_context.num_entries; i++){

        static const char *frame_types = " ipbs";

        char tmp[256];

        RateControlEntry *rce;



        rce= &s->rc_context.entry[i];



        snprintf(tmp, sizeof(tmp), "%c %d %d %d %d %d %d\n",

            frame_types[rce->pict_type], (int)lrintf(rce->qscale / FF_QP2LAMBDA), rce->i_count, s->mb_num - rce->i_count - rce->skip_count,

            rce->skip_count, (rce->i_tex_bits + rce->p_tex_bits + rce->misc_bits+7)/8, (rce->header_bits+rce->mv_bits+7)/8);



//av_log(NULL, AV_LOG_ERROR, "%s\n", tmp);

        write(fd, tmp, strlen(tmp));

    }



    close(fd);



    memset(&xvid_2pass2, 0, sizeof(xvid_2pass2));

    xvid_2pass2.version= XVID_MAKE_VERSION(1,1,0);

    xvid_2pass2.filename= tmp_name;

    xvid_2pass2.bitrate= s->avctx->bit_rate;

    xvid_2pass2.vbv_size= s->avctx->rc_buffer_size;

    xvid_2pass2.vbv_maxrate= s->avctx->rc_max_rate;

    xvid_2pass2.vbv_initial= s->avctx->rc_initial_buffer_occupancy;



    memset(&xvid_plg_create, 0, sizeof(xvid_plg_create));

    xvid_plg_create.version= XVID_MAKE_VERSION(1,1,0);

    xvid_plg_create.fbase= s->avctx->time_base.den;

    xvid_plg_create.fincr= s->avctx->time_base.num;

    xvid_plg_create.param= &xvid_2pass2;



    if(xvid_plugin_2pass2(NULL, XVID_PLG_CREATE, &xvid_plg_create, &s->rc_context.non_lavc_opaque)<0){

        av_log(NULL, AV_LOG_ERROR, "xvid_plugin_2pass2 failed\n");

        return -1;

    }

    return 0;

}
