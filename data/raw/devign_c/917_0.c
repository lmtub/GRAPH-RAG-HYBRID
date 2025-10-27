void avcodec_get_channel_layout_string(char *buf, int buf_size, int nb_channels, int64_t channel_layout)

{

    int i;



    if (channel_layout==0)

        channel_layout = avcodec_guess_channel_layout(nb_channels, CODEC_ID_NONE, NULL);



    for (i=0; channel_layout_map[i].name; i++)

        if (nb_channels    == channel_layout_map[i].nb_channels &&

            channel_layout == channel_layout_map[i].layout) {

            snprintf(buf, buf_size, channel_layout_map[i].name);

            return;

        }



    snprintf(buf, buf_size, "%d channels", nb_channels);

    if (channel_layout) {

        int i,ch;

        av_strlcat(buf, " (", buf_size);

        for(i=0,ch=0; i<64; i++) {

            if ((channel_layout & (1L<<i))) {

                const char *name = get_channel_name(i);

                if (name) {

                    if (ch>0) av_strlcat(buf, "|", buf_size);

                    av_strlcat(buf, name, buf_size);

                }

                ch++;

            }

        }

        av_strlcat(buf, ")", buf_size);

    }

}
