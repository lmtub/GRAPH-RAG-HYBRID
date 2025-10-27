int64_t throttle_compute_wait(LeakyBucket *bkt)

{

    double extra; /* the number of extra units blocking the io */

    double bucket_size;   /* I/O before throttling to bkt->avg */

    double burst_bucket_size; /* Before throttling to bkt->max */



    if (!bkt->avg) {

        return 0;

    }



    if (!bkt->max) {

        /* If bkt->max is 0 we still want to allow short bursts of I/O

         * from the guest, otherwise every other request will be throttled

         * and performance will suffer considerably. */

        bucket_size = (double) bkt->avg / 10;

        burst_bucket_size = 0;

    } else {

        /* If we have a burst limit then we have to wait until all I/O

         * at burst rate has finished before throttling to bkt->avg */

        bucket_size = bkt->max * bkt->burst_length;

        burst_bucket_size = (double) bkt->max / 10;

    }



    /* If the main bucket is full then we have to wait */

    extra = bkt->level - bucket_size;

    if (extra > 0) {

        return throttle_do_compute_wait(bkt->avg, extra);

    }



    /* If the main bucket is not full yet we still have to check the

     * burst bucket in order to enforce the burst limit */

    if (bkt->burst_length > 1) {


        extra = bkt->burst_level - burst_bucket_size;

        if (extra > 0) {

            return throttle_do_compute_wait(bkt->max, extra);

        }

    }



    return 0;

}