static void free_input_threads(void)

{

    int i;



    if (nb_input_files == 1)

        return;



    transcoding_finished = 1;



    for (i = 0; i < nb_input_files; i++) {

        InputFile *f = input_files[i];

        AVPacket pkt;



        if (f->joined)

            continue;



        pthread_mutex_lock(&f->fifo_lock);

        while (av_fifo_size(f->fifo)) {

            av_fifo_generic_read(f->fifo, &pkt, sizeof(pkt), NULL);

            av_free_packet(&pkt);

        }

        pthread_cond_signal(&f->fifo_cond);

        pthread_mutex_unlock(&f->fifo_lock);



        pthread_join(f->thread, NULL);

        f->joined = 1;



        while (av_fifo_size(f->fifo)) {

            av_fifo_generic_read(f->fifo, &pkt, sizeof(pkt), NULL);

            av_free_packet(&pkt);

        }

        av_fifo_free(f->fifo);

    }

}
