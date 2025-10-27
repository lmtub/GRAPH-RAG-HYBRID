static int vorbis_encode_frame(AVCodecContext *avccontext,

                               unsigned char *packets,

                               int buf_size, void *data)

{

    vorbis_enc_context *venc = avccontext->priv_data;

    const signed short *audio = data;

    int samples = data ? avccontext->frame_size : 0;

    vorbis_enc_mode *mode;

    vorbis_enc_mapping *mapping;

    PutBitContext pb;

    int i;



    if (!apply_window_and_mdct(venc, audio, samples))

        return 0;

    samples = 1 << (venc->log2_blocksize[0] - 1);



    init_put_bits(&pb, packets, buf_size);



    put_bits(&pb, 1, 0); // magic bit



    put_bits(&pb, ilog(venc->nmodes - 1), 0); // 0 bits, the mode



    mode    = &venc->modes[0];

    mapping = &venc->mappings[mode->mapping];

    if (mode->blockflag) {

        put_bits(&pb, 1, 0);

        put_bits(&pb, 1, 0);

    }



    for (i = 0; i < venc->channels; i++) {

        vorbis_enc_floor *fc = &venc->floors[mapping->floor[mapping->mux[i]]];

        uint16_t posts[MAX_FLOOR_VALUES];

        floor_fit(venc, fc, &venc->coeffs[i * samples], posts, samples);

        floor_encode(venc, fc, &pb, posts, &venc->floor[i * samples], samples);

    }



    for (i = 0; i < venc->channels * samples; i++)

        venc->coeffs[i] /= venc->floor[i];



    for (i = 0; i < mapping->coupling_steps; i++) {

        float *mag = venc->coeffs + mapping->magnitude[i] * samples;

        float *ang = venc->coeffs + mapping->angle[i]     * samples;

        int j;

        for (j = 0; j < samples; j++) {

            float a = ang[j];

            ang[j] -= mag[j];

            if (mag[j] > 0)

                ang[j] = -ang[j];

            if (ang[j] < 0)

                mag[j] = a;

        }

    }



    residue_encode(venc, &venc->residues[mapping->residue[mapping->mux[0]]],

                   &pb, venc->coeffs, samples, venc->channels);



    avccontext->coded_frame->pts = venc->sample_count;

    venc->sample_count += avccontext->frame_size;

    flush_put_bits(&pb);

    return put_bits_count(&pb) >> 3;

}
