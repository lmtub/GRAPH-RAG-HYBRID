static int hdcd_envelope(int32_t *samples, int count, int stride, int gain, int target_gain, int extend)

{

    int i;

    int32_t *samples_end = samples + stride * count;



    if (extend) {

        for (i = 0; i < count; i++) {

            int32_t sample = samples[i * stride];

            int32_t asample = abs(sample) - 0x5981;

            if (asample >= 0)

                sample = sample >= 0 ? peaktab[asample] : -peaktab[asample];

            else

                sample <<= 15;



            samples[i * stride] = sample;

        }

    } else {

        for (i = 0; i < count; i++)

            samples[i * stride] <<= 15;

    }



    if (gain <= target_gain) {

        int len = FFMIN(count, target_gain - gain);

        /* attenuate slowly */

        for (i = 0; i < len; i++) {

            ++gain;

            APPLY_GAIN(*samples, gain);

            samples += stride;

        }

        count -= len;

    } else {

        int len = FFMIN(count, (gain - target_gain) >> 3);

        /* amplify quickly */

        for (i = 0; i < len; i++) {

            gain -= 8;

            APPLY_GAIN(*samples, gain);

            samples += stride;

        }

        if (gain - 8 < target_gain)

            gain = target_gain;

        count -= len;

    }



    /* hold a steady level */

    if (gain == 0) {

        if (count > 0)

            samples += count * stride;

    } else {

        while (--count >= 0) {

            APPLY_GAIN(*samples, gain);

            samples += stride;

        }

    }



    av_assert0(samples == samples_end);



    return gain;

}
