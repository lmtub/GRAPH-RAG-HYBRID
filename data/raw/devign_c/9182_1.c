static int compute_mask(int step, uint32_t *mask)

{

    int i, z, ret = 0;

    int counter_size = sizeof(uint32_t) * (2 * step + 1);

    uint32_t *temp1_counter, *temp2_counter, **counter;

    temp1_counter = av_mallocz(counter_size);

    if (!temp1_counter) {

        ret = AVERROR(ENOMEM);

        goto end;

    }

    temp2_counter = av_mallocz(counter_size);

    if (!temp2_counter) {

        ret = AVERROR(ENOMEM);

        goto end;

    }

    counter = av_mallocz_array(2 * step + 1, sizeof(uint32_t *));

    if (!counter) {

        ret = AVERROR(ENOMEM);

        goto end;

    }

    for (i = 0; i < 2 * step + 1; i++) {

        counter[i] = av_mallocz(counter_size);

        if (!counter[i]) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

    }

    for (i = 0; i < 2 * step + 1; i++) {

        memset(temp1_counter, 0, counter_size);

        temp1_counter[i] = 1;

        for (z = 0; z < step * 2; z += 2) {

            add_mask_counter(temp2_counter, counter[z], temp1_counter, step * 2);

            memcpy(counter[z], temp1_counter, counter_size);

            add_mask_counter(temp1_counter, counter[z + 1], temp2_counter, step * 2);

            memcpy(counter[z + 1], temp2_counter, counter_size);

        }

    }

    memcpy(mask, temp1_counter, counter_size);

end:

    av_freep(&temp1_counter);

    av_freep(&temp2_counter);

    for (i = 0; i < 2 * step + 1; i++) {

        av_freep(&counter[i]);

    }

    av_freep(&counter);

    return ret;

}
