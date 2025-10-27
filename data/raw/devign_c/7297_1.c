static int hls_window(AVFormatContext *s, int last)

{

    HLSContext *hls = s->priv_data;

    HLSSegment *en;

    int target_duration = 0;

    int ret = 0;

    AVIOContext *out = NULL;

    AVIOContext *sub_out = NULL;

    char temp_filename[1024];

    int64_t sequence = FFMAX(hls->start_sequence, hls->sequence - hls->nb_entries);

    int version = 3;

    const char *proto = avio_find_protocol_name(s->filename);

    int use_rename = proto && !strcmp(proto, "file");

    static unsigned warned_non_file;

    char *key_uri = NULL;

    char *iv_string = NULL;

    AVDictionary *options = NULL;

    double prog_date_time = hls->initial_prog_date_time;

    int byterange_mode = (hls->flags & HLS_SINGLE_FILE) || (hls->max_seg_size > 0);



    if (byterange_mode) {

        version = 4;

        sequence = 0;

    }



    if (hls->segment_type == SEGMENT_TYPE_FMP4) {

        version = 7;

    }



    if (!use_rename && !warned_non_file++)

        av_log(s, AV_LOG_ERROR, "Cannot use rename on non file protocol, this may lead to races and temporary partial files\n");



    set_http_options(s, &options, hls);

    snprintf(temp_filename, sizeof(temp_filename), use_rename ? "%s.tmp" : "%s", s->filename);

    if ((ret = s->io_open(s, &out, temp_filename, AVIO_FLAG_WRITE, &options)) < 0)

        goto fail;



    for (en = hls->segments; en; en = en->next) {

        if (target_duration <= en->duration)

            target_duration = get_int_from_double(en->duration);

    }



    hls->discontinuity_set = 0;

    write_m3u8_head_block(hls, out, version, target_duration, sequence);

    if (hls->pl_type == PLAYLIST_TYPE_EVENT) {

        avio_printf(out, "#EXT-X-PLAYLIST-TYPE:EVENT\n");

    } else if (hls->pl_type == PLAYLIST_TYPE_VOD) {

        avio_printf(out, "#EXT-X-PLAYLIST-TYPE:VOD\n");

    }



    if((hls->flags & HLS_DISCONT_START) && sequence==hls->start_sequence && hls->discontinuity_set==0 ){

        avio_printf(out, "#EXT-X-DISCONTINUITY\n");

        hls->discontinuity_set = 1;

    }

    for (en = hls->segments; en; en = en->next) {

        if ((hls->encrypt || hls->key_info_file) && (!key_uri || strcmp(en->key_uri, key_uri) ||

                                    av_strcasecmp(en->iv_string, iv_string))) {

            avio_printf(out, "#EXT-X-KEY:METHOD=AES-128,URI=\"%s\"", en->key_uri);

            if (*en->iv_string)

                avio_printf(out, ",IV=0x%s", en->iv_string);

            avio_printf(out, "\n");

            key_uri = en->key_uri;

            iv_string = en->iv_string;

        }



        if (en->discont) {

            avio_printf(out, "#EXT-X-DISCONTINUITY\n");

        }



        if ((hls->segment_type == SEGMENT_TYPE_FMP4) && (en == hls->segments)) {

            avio_printf(out, "#EXT-X-MAP:URI=\"%s\"", hls->fmp4_init_filename);

            if (hls->flags & HLS_SINGLE_FILE) {

                avio_printf(out, ",BYTERANGE=\"%"PRId64"@%"PRId64"\"", en->size, en->pos);

            }

            avio_printf(out, "\n");

        } else {

            if (hls->flags & HLS_ROUND_DURATIONS)

                avio_printf(out, "#EXTINF:%ld,\n",  lrint(en->duration));

            else

                avio_printf(out, "#EXTINF:%f,\n", en->duration);

            if (byterange_mode)

                avio_printf(out, "#EXT-X-BYTERANGE:%"PRId64"@%"PRId64"\n",

                            en->size, en->pos);

        }

        if (hls->flags & HLS_PROGRAM_DATE_TIME) {

            time_t tt, wrongsecs;

            int milli;

            struct tm *tm, tmpbuf;

            char buf0[128], buf1[128];

            tt = (int64_t)prog_date_time;

            milli = av_clip(lrint(1000*(prog_date_time - tt)), 0, 999);

            tm = localtime_r(&tt, &tmpbuf);

            strftime(buf0, sizeof(buf0), "%Y-%m-%dT%H:%M:%S", tm);

            if (!strftime(buf1, sizeof(buf1), "%z", tm) || buf1[1]<'0' ||buf1[1]>'2') {

                int tz_min, dst = tm->tm_isdst;

                tm = gmtime_r(&tt, &tmpbuf);

                tm->tm_isdst = dst;

                wrongsecs = mktime(tm);

                tz_min = (abs(wrongsecs - tt) + 30) / 60;

                snprintf(buf1, sizeof(buf1),

                         "%c%02d%02d",

                         wrongsecs <= tt ? '+' : '-',

                         tz_min / 60,

                         tz_min % 60);

            }

            avio_printf(out, "#EXT-X-PROGRAM-DATE-TIME:%s.%03d%s\n", buf0, milli, buf1);

            prog_date_time += en->duration;

        }

        if (!((hls->segment_type == SEGMENT_TYPE_FMP4) && (en == hls->segments))) {

            if (hls->baseurl)

                avio_printf(out, "%s", hls->baseurl);

            avio_printf(out, "%s\n", en->filename);

        }

    }



    if (last && (hls->flags & HLS_OMIT_ENDLIST)==0)

        avio_printf(out, "#EXT-X-ENDLIST\n");



    if( hls->vtt_m3u8_name ) {

        if ((ret = s->io_open(s, &sub_out, hls->vtt_m3u8_name, AVIO_FLAG_WRITE, &options)) < 0)

            goto fail;

        write_m3u8_head_block(hls, sub_out, version, target_duration, sequence);



        for (en = hls->segments; en; en = en->next) {

            avio_printf(sub_out, "#EXTINF:%f,\n", en->duration);

            if (byterange_mode)

                 avio_printf(sub_out, "#EXT-X-BYTERANGE:%"PRIi64"@%"PRIi64"\n",

                         en->size, en->pos);

            if (hls->baseurl)

                avio_printf(sub_out, "%s", hls->baseurl);

            avio_printf(sub_out, "%s\n", en->sub_filename);

        }



        if (last)

            avio_printf(sub_out, "#EXT-X-ENDLIST\n");



    }



fail:

    av_dict_free(&options);

    ff_format_io_close(s, &out);

    ff_format_io_close(s, &sub_out);

    if (ret >= 0 && use_rename)

        ff_rename(temp_filename, s->filename, s);

    return ret;

}
