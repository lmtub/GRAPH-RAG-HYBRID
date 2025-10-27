static void move_audio(vorbis_enc_context *venc, float **audio, int *samples, int sf_size)

{

    AVFrame *cur = NULL;

    int frame_size = 1 << (venc->log2_blocksize[1] - 1);

    int subframes = frame_size / sf_size;



    for (int sf = 0; sf < subframes; sf++) {

        cur = ff_bufqueue_get(&venc->bufqueue);

        *samples += cur->nb_samples;



        for (int ch = 0; ch < venc->channels; ch++) {

            const float *input = (float *) cur->extended_data[ch];

            const size_t len  = cur->nb_samples * sizeof(float);

            memcpy(&audio[ch][sf*sf_size], input, len);

        }

        av_frame_free(&cur);

    }

}
