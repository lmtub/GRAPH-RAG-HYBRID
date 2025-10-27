static void update_md5_sum(FlacEncodeContext *s, const int16_t *samples)

{

#if HAVE_BIGENDIAN

    int i;

    for (i = 0; i < s->frame.blocksize * s->channels; i++) {

        int16_t smp = av_le2ne16(samples[i]);

        av_md5_update(s->md5ctx, (uint8_t *)&smp, 2);

    }

#else

    av_md5_update(s->md5ctx, (const uint8_t *)samples, s->frame.blocksize*s->channels*2);

#endif

}
